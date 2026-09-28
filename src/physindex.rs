//! Physical memory index: maps a page-aligned [`Gpa`] to the file offset where
//! the content of that page lives.
//!
//! Two strategies are available:
//! - [`PhysIndexStrategy::Eager`]: one map entry per page. Memory grows with the
//!   number of pages in the dump.
//! - [`PhysIndexStrategy::Compact`]: sorted runs of consecutive pages (full,
//!   kernel and complete dumps) or the dump bitmap plus a rank table (bitmap
//!   dumps). Memory grows with the number of runs or with the size of the
//!   bitmap instead of with the number of pages.
use crate::error::{Error, Result};
use crate::gxa::{Gpa, Gxa};
use crate::pxe::Pfn;
use crate::structs::{PageKind, PhysmemMap};

/// How the parser indexes the physical memory found in the dump.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PhysIndexStrategy {
    /// One entry per page.
    Eager,
    /// Runs of consecutive pages, or the bitmap plus a rank table.
    #[default]
    Compact,
}

/// Number of bits (pages) covered by one rank-table entry.
const RANK_BLOCK_BITS: u64 = 4_096;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Run {
    first_pfn: u64,
    pages: u64,
    file_offset: u64,
}

pub(crate) enum PhysIndex {
    Eager(PhysmemMap),
    Runs {
        runs: Vec<Run>,
        pages: u64,
    },
    Bitmap {
        first_page_offset: u64,
        bits: Vec<u8>,
        rank: Vec<u64>,
        pages: u64,
    },
}

/// Collects pages in file order; coalesces consecutive pages into runs when
/// the compact strategy is used.
pub(crate) struct Builder {
    strategy: PhysIndexStrategy,
    map: PhysmemMap,
    runs: Vec<Run>,
    pages: u64,
}

impl Builder {
    pub fn new(strategy: PhysIndexStrategy) -> Self {
        Self {
            strategy,
            map: PhysmemMap::new(),
            runs: Vec::new(),
            pages: 0,
        }
    }

    /// Records a page. Returns `false` if the eager map already had it.
    pub fn push(&mut self, gpa: Gpa, file_offset: u64) -> bool {
        self.pages += 1;
        match self.strategy {
            PhysIndexStrategy::Eager => self.map.insert(gpa, file_offset).is_none(),
            PhysIndexStrategy::Compact => {
                let pfn = gpa.u64() / PageKind::Normal.size();
                if let Some(last) = self.runs.last_mut()
                    && last.first_pfn + last.pages == pfn
                    && last.file_offset + last.pages * PageKind::Normal.size() == file_offset
                {
                    last.pages += 1;
                    return true;
                }

                self.runs.push(Run {
                    first_pfn: pfn,
                    pages: 1,
                    file_offset,
                });

                true
            }
        }
    }

    /// Records `pages` consecutive pages starting at `gpa` whose contents are
    /// stored back to back from `file_offset`. Returns `false` if the eager
    /// map already had one of them.
    pub fn push_run(&mut self, gpa: Gpa, pages: u64, file_offset: u64) -> bool {
        if pages == 0 {
            return true;
        }

        match self.strategy {
            PhysIndexStrategy::Eager => {
                let size = PageKind::Normal.size();
                (0..pages)
                    .all(|i| self.push(Gpa::new(gpa.u64() + i * size), file_offset + i * size))
            }
            PhysIndexStrategy::Compact => {
                self.push(gpa, file_offset);
                let last = self.runs.last_mut().unwrap();
                last.pages += pages - 1;
                self.pages += pages - 1;

                true
            }
        }
    }

    /// Finalizes the index. Overlapping runs are reported like duplicate
    /// pages in the eager map.
    pub fn finish(mut self) -> Result<PhysIndex> {
        match self.strategy {
            PhysIndexStrategy::Eager => Ok(PhysIndex::Eager(self.map)),
            PhysIndexStrategy::Compact => {
                self.runs.sort_by_key(|r| r.first_pfn);
                for w in self.runs.windows(2) {
                    if w[0].first_pfn + w[0].pages > w[1].first_pfn {
                        return Err(Error::DuplicateGpa(Pfn::new(w[1].first_pfn).gpa()));
                    }
                }

                Ok(PhysIndex::Runs {
                    runs: self.runs,
                    pages: self.pages,
                })
            }
        }
    }
}

impl PhysIndex {
    /// Builds a bitmap index; `bits` is the dump bitmap with the bits past the
    /// page count cleared.
    pub fn bitmap(first_page_offset: u64, bits: Vec<u8>) -> Self {
        let block_bytes = usize::try_from(RANK_BLOCK_BITS / 8).unwrap();
        let mut rank = Vec::with_capacity(bits.len() / block_bytes + 1);
        let mut total = 0u64;
        for chunk in bits.chunks(block_bytes) {
            rank.push(total);
            total += chunk.iter().map(|b| u64::from(b.count_ones())).sum::<u64>();
        }

        PhysIndex::Bitmap {
            first_page_offset,
            bits,
            rank,
            pages: total,
        }
    }

    /// File offset of the page at `page`, if the dump has it.
    pub fn get(&self, page: Gpa) -> Option<u64> {
        let size = PageKind::Normal.size();
        match self {
            PhysIndex::Eager(m) => m.get(&page).copied(),
            PhysIndex::Runs { runs, .. } => {
                let pfn = page.u64() / size;
                let i = runs
                    .partition_point(|r| r.first_pfn <= pfn)
                    .checked_sub(1)?;
                let r = runs[i];
                (pfn < r.first_pfn + r.pages).then(|| r.file_offset + (pfn - r.first_pfn) * size)
            }
            PhysIndex::Bitmap {
                first_page_offset,
                bits,
                rank,
                ..
            } => {
                let pfn = page.u64() / size;
                let byte = usize::try_from(pfn / 8).ok()?;
                let bit = u32::try_from(pfn % 8).ok()?;
                if *bits.get(byte)? & (1 << bit) == 0 {
                    return None;
                }

                let blk = usize::try_from(pfn / RANK_BLOCK_BITS).ok()?;
                let start = blk * usize::try_from(RANK_BLOCK_BITS / 8).ok()?;
                let before = rank[blk]
                    + bits[start..byte]
                        .iter()
                        .map(|b| u64::from(b.count_ones()))
                        .sum::<u64>()
                    + u64::from((bits[byte] & ((1u8 << bit) - 1)).count_ones());

                Some(first_page_offset + before * size)
            }
        }
    }

    /// Number of pages in the index.
    pub fn len(&self) -> usize {
        match self {
            PhysIndex::Eager(m) => m.len(),
            PhysIndex::Runs { pages, .. } | PhysIndex::Bitmap { pages, .. } => {
                usize::try_from(*pages).unwrap_or(usize::MAX)
            }
        }
    }

    /// Iterates over (page, file offset) pairs in ascending page order.
    pub fn iter(&self) -> PhysIter<'_> {
        let size = PageKind::Normal.size();
        let inner: Box<dyn Iterator<Item = (Gpa, u64)> + '_> = match self {
            PhysIndex::Eager(m) => Box::new(m.iter().map(|(&k, &v)| (k, v))),
            PhysIndex::Runs { runs, .. } => Box::new(runs.iter().flat_map(move |r| {
                (0..r.pages)
                    .map(move |i| (Pfn::new(r.first_pfn + i).gpa(), r.file_offset + i * size))
            })),
            PhysIndex::Bitmap {
                first_page_offset,
                bits,
                ..
            } => {
                let base = *first_page_offset;
                Box::new(
                    bits.iter()
                        .enumerate()
                        .flat_map(|(i, b)| {
                            (0..8u64)
                                .filter(move |k| b & (1 << k) != 0)
                                .map(move |k| i as u64 * 8 + k)
                        })
                        .zip(0u64..)
                        .map(move |(pfn, n)| (Pfn::new(pfn).gpa(), base + n * size)),
                )
            }
        };

        PhysIter {
            inner,
            left: self.len(),
        }
    }
}

/// Iterator over (page, file offset) pairs with an exact length.
pub struct PhysIter<'a> {
    inner: Box<dyn Iterator<Item = (Gpa, u64)> + 'a>,
    left: usize,
}

impl Iterator for PhysIter<'_> {
    type Item = (Gpa, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let v = self.inner.next()?;
        self.left -= 1;

        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for PhysIter<'_> {}
