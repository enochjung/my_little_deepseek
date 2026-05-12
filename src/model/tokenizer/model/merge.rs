use crate::config::{Format, MergeFormat};
use std::collections::{HashMap, LinkedList};

pub(crate) struct MergeEngine {
    merge_map: MergeMap,
}

impl MergeEngine {
    pub(crate) fn new(merge_format: &MergeFormat) -> Result<Self, crate::Error> {
        let mut merge_map = MergeMap::new();

        let mut rank = 0;
        let (file, parse) = merge_format.read()?;
        let iter = parse(&file);

        for line in iter {
            let line = line?;
            let (left, right) = line;
            let key = Key::new(&left, &right);
            merge_map.insert(key, rank);
            rank += 1;
        }

        Ok(Self { merge_map })
    }

    pub(crate) fn merge(&self, list: &[String]) -> Result<Vec<String>, crate::Error> {
        let mut tokens: LinkedList<String> = list.iter().cloned().collect();

        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            let mut prev = tokens.front().unwrap();
            let mut idx = 0;
            for token in tokens.iter().skip(1) {
                let key = Key::new(prev, token);
                if let Some(rank) = self.merge_map.get(&key) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = idx;
                    }
                }

                prev = token;
                idx += 1;
            }

            if best_rank == usize::MAX {
                break;
            }

            let mut tail = tokens.split_off(best_idx);
            let left = tail.pop_front().unwrap();
            let right = tail.pop_front().unwrap();
            let merged = format!("{}{}", left, right);

            tokens.push_back(merged);
            tokens.append(&mut tail);
        }

        Ok(tokens.into_iter().collect())
    }
}

struct MergeMap {
    map: HashMap<Key, usize>,
}

impl MergeMap {
    fn new() -> Self {
        Self {
            map: HashMap::with_capacity(151388),
        }
    }

    fn get(&self, key: &Key) -> Option<usize> {
        self.map.get(key).copied()
    }

    fn insert(&mut self, key: Key, rank: usize) -> () {
        self.map.insert(key, rank);
    }
}

#[derive(Eq, Hash, PartialEq)]
struct Key {
    // "left"\x1e"right"
    key: Vec<u8>,
}

impl Key {
    fn new(left: &str, right: &str) -> Self {
        let mut key = left.as_bytes().to_vec();
        key.push(0x1e);
        key.extend(right.as_bytes());

        Self { key }
    }
}
