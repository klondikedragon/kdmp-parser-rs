//! Physical memory index: maps a page-aligned [`Gpa`] to the file offset where
//! the content of that page lives. Sorted runs of consecutive pages (full,
//! kernel and complete dumps) or the dump bitmap plus a rank table (bitmap
//! dumps). Memory grows with the number of runs or with the size of the
//! bitmap instead of with the number of pages.
use std::num::NonZeroU64;
use std::ops::Range;

use crate::error::{Error, Result};
use crate::gxa::{Gpa, Gxa};
use crate::pxe::Pfn;
use crate::structs::PageKind;

/// Number of bits (pages) covered by one rank-table entry.
const RANK_BLOCK_BITS: u64 = 4_096;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Run {
    first_pfn: Pfn,
    pages: u64,
    file_offset: u64,
}

impl Run {
    pub(crate) fn pfn_range(&self) -> Range<Pfn> {
        let start = self.first_pfn;
        let end = start + self.pages;
        debug_assert!(end > start);

        start..end
    }

    pub(crate) fn file_offset_range(&self) -> Range<u64> {
        let start = self.file_offset;
        let len = self.pages * PageKind::Normal.len();
        let end = start + len;
        debug_assert!(end > start);

        start..end
    }

    pub(crate) fn is_contiguous_after(&self, other: &Run) -> bool {
        // For two runs to be contiguous, they need to have contiguous physical
        // addresses as well as being stored in the dump file with contiguous
        // file offsets.
        self.pfn_range().start == other.pfn_range().end
            && self.file_offset_range().start == other.file_offset_range().end
    }

    pub(crate) fn file_offset_for_pfn(&self, pfn: Pfn) -> Option<u64> {
        let page_index = pfn.u64().checked_sub(self.first_pfn.u64())?;
        if page_index >= self.pages {
            return None;
        }

        let offset_from_run = page_index.checked_mul(PageKind::Normal.len())?;

        self.file_offset.checked_add(offset_from_run)
    }
}

/// Collects pages in file order; coalesces consecutive pages into runs.
pub(crate) struct Builder {
    runs: Vec<Run>,
    total_pages: u64,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            total_pages: 0,
        }
    }

    /// Records a page.
    pub fn push(&mut self, gpa: Gpa, file_offset: u64) {
        self.total_pages += 1;
        let pfn = gpa.pfn();
        let run = Run {
            first_pfn: pfn,
            pages: 1,
            file_offset,
        };

        // Try to merge this page w/ the last run we added if it contiguous with
        // this page..
        if let Some(last) = self.runs.last_mut()
            && run.is_contiguous_after(last)
        {
            last.pages += 1;
        }
        // Otherwise simply create a new run with 1 page.
        else {
            self.runs.push(Run {
                first_pfn: pfn,
                pages: 1,
                file_offset,
            });
        }
    }

    /// Records `pages` consecutive pages starting at `gpa` whose contents are
    /// stored back to back from `file_offset`.
    pub fn push_run(&mut self, gpa: Gpa, pages: NonZeroU64, file_offset: u64) {
        self.push(gpa, file_offset);
        // SAFETY: We can `unwrap` here unconditionally because after the `push`
        // we have at least one runs available.
        let last = self.runs.last_mut().unwrap();
        // The `push` call increased or set the `pages` to 1, so deduct it.
        let pages = pages.get() - 1;
        last.pages += pages;
        self.total_pages += pages;
    }

    /// Finalizes the index.
    pub fn finish(mut self) -> Result<PhysIndex> {
        // Sort the runs.. (we don't care if elements that are equal are
        // re-ordered, so unstability is fine)..
        self.runs.sort_unstable_by_key(|r| r.first_pfn);

        // Detect duplicate gpas / overlaped runs..
        for [run, next_run] in self.runs.array_windows::<2>() {
            if run.pfn_range().end > next_run.pfn_range().start {
                return Err(Error::InvalidData("duplicate gpa in phys index"));
            }
        }

        // Merge every runs we can..
        self.runs.dedup_by(|next, prev| {
            // If `next` is contiguous after `p`, then let's merge `next` into
            // `prev`..
            if next.is_contiguous_after(prev) {
                prev.pages += next.pages;

                // ..and delete `next`.
                true
            }
            // Otherwise keep both runs.
            else {
                false
            }
        });

        // ..and shrink the vec.
        self.runs.shrink_to_fit();

        Ok(PhysIndex::Runs {
            runs: self.runs,
            total_pages: self.total_pages,
        })
    }
}

pub(crate) enum PhysIndex {
    Runs {
        runs: Vec<Run>,
        total_pages: u64,
    },
    Bitmap {
        first_page_offset: u64,
        bits: Vec<u8>,
        rank: Vec<u64>,
        total_pages: u64,
    },
}

impl PhysIndex {
    /// Builds a bitmap index; `bits` is the dump bitmap with the bits past the
    /// page count cleared.
    pub fn bitmap(first_page_offset: u64, bits: Vec<u8>) -> Self {
        let block_bytes = usize::try_from(RANK_BLOCK_BITS / 8).unwrap();
        let mut rank = Vec::with_capacity(bits.len() / block_bytes + 1);
        let mut total_pages = 0u64;
        for chunk in bits.chunks(block_bytes) {
            rank.push(total_pages);
            total_pages += chunk.iter().map(|b| u64::from(b.count_ones())).sum::<u64>();
        }

        PhysIndex::Bitmap {
            first_page_offset,
            bits,
            rank,
            total_pages,
        }
    }

    /// File offset of the page at `page`, if the dump has it.
    pub fn get(&self, page: Gpa) -> Option<u64> {
        let len = PageKind::Normal.len();
        match self {
            PhysIndex::Runs { runs, .. } => {
                // Find the index of the first run that starts at a pfn greater
                // than the one we're looking for. We partition
                // the runs in two; the left side has runs that starts at a pfn
                // smaller or equal, and the right one has runs
                // that start w/ a greater pfn.
                //
                // For example, the `partition_idx` would be:
                //  - `0` if `pfn` was `1`
                //  - `2` if `pfn` was `6`
                //  - `3` if `pfn` was `11`
                //
                // Runs: `[start: 2, end: 5], [start: 5, end: 6], [start: 10,
                // end: 14]`
                let pfn = page.pfn();
                let partition_idx = runs.partition_point(|r| r.pfn_range().start <= pfn);

                // The partition idx is the first item that has a larger start
                // pfn.. which means that our `pfn` is inside the run right
                // before, so we're done.
                if partition_idx == 0 {
                    return None;
                }

                // This run is the one that potentially includes the `pfn` we're
                // after. It is possible to end up in a case like index `2` for
                // the pfn `6` in the above example. The pfn `6` is actually in
                // a hole in between the two sides of the partition.
                let run = runs[partition_idx - 1];

                // Get the file offset if the run actuall this pfn.
                run.file_offset_for_pfn(pfn)
            }
            PhysIndex::Bitmap {
                first_page_offset,
                bits,
                rank,
                ..
            } => {
                let pfn = page.u64() / len;
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

                Some(first_page_offset + before * len)
            }
        }
    }

    /// Number of pages in the index.
    pub fn pages_amount(&self) -> usize {
        match self {
            PhysIndex::Runs { total_pages, .. } | PhysIndex::Bitmap { total_pages, .. } => {
                usize::try_from(*total_pages).unwrap_or(usize::MAX)
            }
        }
    }

    /// Iterates over (page, file offset) pairs in ascending page order.
    pub fn iter(&self) -> PhysIter<'_> {
        let len = PageKind::Normal.len();
        let inner: Box<dyn Iterator<Item = (Gpa, u64)> + '_> = match self {
            PhysIndex::Runs { runs, .. } => Box::new(runs.iter().flat_map(move |r| {
                (0..r.pages).map(move |i| ((r.first_pfn + i).gpa(), r.file_offset + i * len))
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
                        .map(move |(pfn, n)| (Pfn::new(pfn).gpa(), base + n * len)),
                )
            }
        };

        PhysIter {
            inner,
            left: self.pages_amount(),
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use crate::error::Error;
    use crate::gxa::Gpa;
    use crate::physindex::{Builder, PhysIndex, Run};

    fn assert_runs(builder: Builder, expected: &[Run]) {
        let PhysIndex::Runs { runs, total_pages } = builder.finish().unwrap() else {
            panic!("expected a runs index");
        };

        assert_eq!(runs.len(), expected.len());
        assert_eq!(total_pages, expected.iter().map(|e| e.pages).sum());

        for (run, expected) in runs.iter().zip(expected) {
            assert_eq!(run, expected);
        }
    }

    fn run(gpa: u64, pages: u64, file_offset: u64) -> Run {
        Run {
            first_pfn: Gpa::new(gpa).pfn(),
            pages,
            file_offset,
        }
    }

    fn builder(runs: &[Run]) -> Builder {
        let mut builder = Builder::new();
        for run in runs {
            builder.push_run(
                Gpa::from_pfn(run.first_pfn),
                NonZeroU64::new(run.pages).unwrap(),
                run.file_offset,
            );
        }

        builder
    }

    #[test]
    fn runs_empty_and_single_page() {
        assert_runs(builder(&[]), &[]);

        let builder = builder(&[run(0, 1, 0)]);

        assert_runs(builder, &[run(0, 1, 0)]);
    }

    #[test]
    fn runs_merge_runs() {
        let builder = builder(&[
            run(0x1_000, 1, 0),
            run(0x2_000, 1, 0x1_000),
            run(0x3_000, 3, 0x2_000),
            run(0x6_000, 1, 0x5_000),
        ]);

        assert_runs(builder, &[run(0x1_000, 6, 0)]);
    }

    #[test]
    fn runs_merge_out_of_order_runs() {
        let builder = builder(&[
            run(0x3_000, 2, 0x2_000),
            run(0x1_000, 2, 0),
            run(0x5_000, 2, 0x4_000),
        ]);

        assert_runs(builder, &[run(0x1_000, 6, 0)]);
    }

    #[test]
    fn runs_sorts_runs_but_no_merge_across_gaps() {
        let builder = builder(&[
            run(0x4_000, 2, 0x5_000),
            run(0x1_000, 2, 0),
            // 0x1_000-0x2_000 are adjacent GPAs, but not in the offset space
            // where's a gap of a page.
            run(0x3_000, 1, 0x3_000),
        ]);

        assert_runs(builder, &[
            run(0x1_000, 2, 0),
            run(0x3_000, 1, 0x3_000),
            run(0x4_000, 2, 0x5_000),
        ]);
    }

    #[test]
    fn runs_rejects_duplicates_and_overlaps() {
        // Exact dup.
        let exact_dup = [run(0x1_000, 1, 0), run(0x1_000, 1, 0)];
        assert!(matches!(
            builder(&exact_dup).finish(),
            Err(Error::InvalidData(_))
        ));

        // First range fully includes the second one.
        let mut full_inc = [run(0x1_000, 5, 0), run(0x2_000, 2, 0x1_000)];
        assert!(matches!(
            builder(&full_inc).finish(),
            Err(Error::InvalidData(_))
        ));

        // Same as above but reversed order.
        full_inc.reverse();
        assert!(matches!(
            builder(&full_inc).finish(),
            Err(Error::InvalidData(_))
        ));

        // Partial overlap.
        let partial_overlap = [run(0x1_000, 5, 0), run(0x5_000, 2, 0x1_000)];

        assert!(matches!(
            builder(&partial_overlap).finish(),
            Err(Error::InvalidData(_))
        ));
    }
}
