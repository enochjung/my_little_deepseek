use super::table::{Codepoint, CombiningClassMap, CompositionExclusions, CompositionMap};

pub fn recompose(
    ordered: &[Codepoint],
    combining_class_map: &CombiningClassMap,
    composition_map: &CompositionMap,
    composition_exclusions: &CompositionExclusions,
) -> Vec<Codepoint> {
    let mut out = Vec::with_capacity(ordered.len());
    let mut segment: Vec<Codepoint> = Vec::new();

    for &cp in ordered {
        let ccc = combining_class_map.get(cp);
        if ccc == 0 {
            flush_segment(
                &mut out,
                &mut segment,
                combining_class_map,
                composition_map,
                composition_exclusions,
            );
        }
        segment.push(cp);
    }

    flush_segment(
        &mut out,
        &mut segment,
        combining_class_map,
        composition_map,
        composition_exclusions,
    );

    out
}

fn flush_segment(
    out: &mut Vec<Codepoint>,
    segment: &mut Vec<Codepoint>,
    combining_class_map: &CombiningClassMap,
    composition_map: &CompositionMap,
    composition_exclusions: &CompositionExclusions,
) {
    if segment.is_empty() {
        return;
    }

    if combining_class_map.get(segment[0]) != 0 {
        out.extend(segment.iter().copied());
        segment.clear();
        return;
    }

    let mut starter = segment[0];
    let mut kept_marks: Vec<(u8, Codepoint)> = Vec::new();

    for &mark in segment.iter().skip(1) {
        let ccc = combining_class_map.get(mark);
        let blocked = kept_marks
            .last()
            .map(|(prev_ccc, _)| *prev_ccc >= ccc)
            .unwrap_or(false);

        if !blocked {
            if let Some(composed) = composition_map.get(starter, mark) {
                if !composition_exclusions.contains(composed) {
                    starter = composed;
                    continue;
                }
            }
        }

        kept_marks.push((ccc, mark));
    }

    out.push(starter);
    out.extend(kept_marks.into_iter().map(|(_, cp)| cp));
    segment.clear();
}
