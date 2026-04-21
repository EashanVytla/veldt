use std::collections::HashMap;
use veldt_core::FetchUnitId;

/// An indexed binary max-heap keyed by priority (usize).
///
/// Supports O(log n) insert, extract-max, remove, and update (decrease/increase key).
/// The `HashMap<FetchUnitId, usize>` index maps each unit to its position in the
/// heap array, enabling O(1) lookup + O(log n) sift for updates.
#[derive(Debug)]
pub struct IndexedMaxHeap {
    /// Heap array: (FetchUnitId, priority). Max-heap by priority.
    entries: Vec<(FetchUnitId, usize)>,
    /// Maps FetchUnitId -> index in `entries`.
    index: HashMap<FetchUnitId, usize>,
}

impl IndexedMaxHeap {
    pub fn new() -> Self {
        IndexedMaxHeap {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a unit with a priority. If already present, updates the priority.
    pub fn insert(&mut self, unit_id: FetchUnitId, priority: usize) {
        if let Some(&pos) = self.index.get(&unit_id) {
            // Update existing
            let old_priority = self.entries[pos].1;
            self.entries[pos].1 = priority;
            if priority > old_priority {
                self.sift_up(pos);
            } else {
                self.sift_down(pos);
            }
        } else {
            // New entry at the end
            let pos = self.entries.len();
            self.entries.push((unit_id, priority));
            self.index.insert(unit_id, pos);
            self.sift_up(pos);
        }
    }

    /// Update the priority of an existing unit. No-op if not present.
    pub fn update(&mut self, unit_id: FetchUnitId, new_priority: usize) {
        if let Some(&pos) = self.index.get(&unit_id) {
            let old_priority = self.entries[pos].1;
            self.entries[pos].1 = new_priority;
            if new_priority > old_priority {
                self.sift_up(pos);
            } else {
                self.sift_down(pos);
            }
        }
    }

    /// Extract the unit with the maximum priority.
    pub fn extract_max(&mut self) -> Option<(FetchUnitId, usize)> {
        if self.entries.is_empty() {
            return None;
        }

        let last = self.entries.len() - 1;
        self.swap(0, last);

        let (unit_id, priority) = self.entries.pop().unwrap();
        self.index.remove(&unit_id);

        if !self.entries.is_empty() {
            self.sift_down(0);
        }

        Some((unit_id, priority))
    }

    /// Remove a specific unit from the heap. Returns its priority if found.
    pub fn remove(&mut self, unit_id: &FetchUnitId) -> Option<usize> {
        let pos = *self.index.get(unit_id)?;
        let last = self.entries.len() - 1;

        if pos == last {
            let (uid, priority) = self.entries.pop().unwrap();
            self.index.remove(&uid);
            return Some(priority);
        }

        self.swap(pos, last);
        let (uid, priority) = self.entries.pop().unwrap();
        self.index.remove(&uid);

        // Sift the swapped element to its correct position
        if pos < self.entries.len() {
            self.sift_up(pos);
            self.sift_down(pos);
        }

        Some(priority)
    }

    /// Peek at the max element without removing it.
    pub fn peek_max(&self) -> Option<(FetchUnitId, usize)> {
        self.entries.first().copied()
    }

    fn parent(pos: usize) -> usize {
        (pos.wrapping_sub(1)) / 2
    }

    fn left_child(pos: usize) -> usize {
        2 * pos + 1
    }

    fn right_child(pos: usize) -> usize {
        2 * pos + 2
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.entries.swap(a, b);
        let uid_a = self.entries[a].0;
        let uid_b = self.entries[b].0;
        self.index.insert(uid_a, a);
        self.index.insert(uid_b, b);
    }

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = Self::parent(pos);
            if self.entries[pos].1 > self.entries[parent].1 {
                self.swap(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        let len = self.entries.len();
        loop {
            let left = Self::left_child(pos);
            let right = Self::right_child(pos);
            let mut largest = pos;

            if left < len && self.entries[left].1 > self.entries[largest].1 {
                largest = left;
            }
            if right < len && self.entries[right].1 > self.entries[largest].1 {
                largest = right;
            }

            if largest != pos {
                self.swap(pos, largest);
                pos = largest;
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_extract_max() {
        let mut heap = IndexedMaxHeap::new();

        heap.insert(FetchUnitId(1), 10);
        heap.insert(FetchUnitId(2), 30);
        heap.insert(FetchUnitId(3), 20);

        let (id, pri) = heap.extract_max().unwrap();
        assert_eq!(id, FetchUnitId(2));
        assert_eq!(pri, 30);

        let (id, pri) = heap.extract_max().unwrap();
        assert_eq!(id, FetchUnitId(3));
        assert_eq!(pri, 20);

        let (id, pri) = heap.extract_max().unwrap();
        assert_eq!(id, FetchUnitId(1));
        assert_eq!(pri, 10);

        assert!(heap.extract_max().is_none());
    }

    #[test]
    fn test_update_priority() {
        let mut heap = IndexedMaxHeap::new();

        heap.insert(FetchUnitId(1), 10);
        heap.insert(FetchUnitId(2), 20);
        heap.insert(FetchUnitId(3), 15);

        // Increase unit 1's priority to 25 (now max)
        heap.update(FetchUnitId(1), 25);

        let (id, pri) = heap.extract_max().unwrap();
        assert_eq!(id, FetchUnitId(1));
        assert_eq!(pri, 25);
    }

    #[test]
    fn test_remove() {
        let mut heap = IndexedMaxHeap::new();

        heap.insert(FetchUnitId(1), 10);
        heap.insert(FetchUnitId(2), 30);
        heap.insert(FetchUnitId(3), 20);

        // Remove the middle element
        let pri = heap.remove(&FetchUnitId(3));
        assert_eq!(pri, Some(20));
        assert_eq!(heap.len(), 2);

        // Max should still be unit 2
        let (id, _) = heap.extract_max().unwrap();
        assert_eq!(id, FetchUnitId(2));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut heap = IndexedMaxHeap::new();
        heap.insert(FetchUnitId(1), 10);
        assert!(heap.remove(&FetchUnitId(99)).is_none());
    }

    #[test]
    fn test_peek_max() {
        let mut heap = IndexedMaxHeap::new();
        assert!(heap.peek_max().is_none());

        heap.insert(FetchUnitId(1), 10);
        heap.insert(FetchUnitId(2), 20);

        let (id, pri) = heap.peek_max().unwrap();
        assert_eq!(id, FetchUnitId(2));
        assert_eq!(pri, 20);
        assert_eq!(heap.len(), 2); // peek doesn't remove
    }

    #[test]
    fn test_many_elements() {
        let mut heap = IndexedMaxHeap::new();

        // Insert 100 elements in random-ish order
        for i in (0..100).rev() {
            heap.insert(FetchUnitId(i), i as usize);
        }

        // Extract should come out in descending order
        let mut prev = usize::MAX;
        while let Some((_id, pri)) = heap.extract_max() {
            assert!(pri <= prev);
            prev = pri;
        }
    }

    #[test]
    fn test_duplicate_insert_updates() {
        let mut heap = IndexedMaxHeap::new();

        heap.insert(FetchUnitId(1), 10);
        heap.insert(FetchUnitId(1), 20); // should update, not duplicate

        assert_eq!(heap.len(), 1);
        let (id, pri) = heap.extract_max().unwrap();
        assert_eq!(id, FetchUnitId(1));
        assert_eq!(pri, 20);
    }
}
