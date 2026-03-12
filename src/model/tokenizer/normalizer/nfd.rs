use super::table::{Codepoint, CombiningClassMap, DecompositionMap};

pub fn canonical_decompose(
    input: &[Codepoint],
    decomposition_map: &DecompositionMap,
) -> Vec<Codepoint> {
    let mut out = Vec::new();
    for &codepoint in input {
        decompose_codepoint(codepoint, decomposition_map, &mut out);
    }
    out
}

pub fn reorder_by_canonical_class(
    decomposed: &[Codepoint],
    combining_class_map: &CombiningClassMap,
) -> Vec<Codepoint> {
    let mut out = Vec::with_capacity(decomposed.len());
    let mut segment: Vec<Codepoint> = Vec::new();

    for &cp in decomposed {
        let ccc = combining_class_map.get(cp);
        if ccc == 0 {
            flush_segment(&mut out, &mut segment, combining_class_map);
        }
        segment.push(cp);
    }

    flush_segment(&mut out, &mut segment, combining_class_map);
    out
}

fn decompose_codepoint(
    codepoint: Codepoint,
    decomposition_map: &DecompositionMap,
    out: &mut Vec<Codepoint>,
) {
    if let Some(mapped) = decomposition_map.get(codepoint) {
        for &next in mapped {
            decompose_codepoint(next, decomposition_map, out);
        }
        return;
    }

    out.push(codepoint);
}

fn flush_segment(
    out: &mut Vec<Codepoint>,
    segment: &mut Vec<Codepoint>,
    combining_class_map: &CombiningClassMap,
) {
    if segment.is_empty() {
        return;
    }

    let starter = segment[0];
    out.push(starter);

    if segment.len() == 1 {
        segment.clear();
        return;
    }

    let mut marks: Vec<(u8, usize, Codepoint)> = segment
        .iter()
        .copied()
        .enumerate()
        .skip(1)
        .map(|(index, cp)| (combining_class_map.get(cp), index, cp))
        .collect();

    // Stable-by-key ordering: sort by canonical combining class, then original index.
    marks.sort_by_key(|(ccc, index, _)| (*ccc, *index));
    for (_, _, cp) in marks {
        out.push(cp);
    }

    segment.clear();
}
