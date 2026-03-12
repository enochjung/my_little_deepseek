# NFC Normalization: A Deep Dive into Unicode Data Files

## Section 1: NFC Normalization Overview

### What is Unicode Normalization?

Unicode allows multiple ways to represent the same visual character. For example, the character "é" (Latin Small Letter E with Acute) can be represented as:
- **Precomposed form**: A single codepoint `U+00E9`
- **Decomposed form**: Two codepoints: `U+0065` (Latin Small Letter E) + `U+0301` (Combining Acute Accent)

Both representations are canonically equivalent and should be treated as the same character by applications. However, this creates problems for:
- String comparison (two visually identical strings might not match byte-for-byte)
- Text search (finding "é" might miss it if stored as "e + combining mark")
- Data validation (different databases might store the same text differently)
- Filename handling (file systems may reject certain decomposition forms)

Unicode Normalization solves this by converting text into a canonical form. This ensures consistent handling across systems.

### The Four Normalization Forms

Unicode defines four normalization forms:

1. **NFD (Decomposed)**: Breaks precomposed characters into base character + combining marks
   - Example: é → e + combining acute accent
   - Use case: Data preservation, detailed text analysis

2. **NFC (Composed)**: Decomposes then recomposes, preferring precomposed forms
   - Example: e + combining acute accent → é
   - Use case: Default choice; most compatible; preferred by W3C

3. **NFKD (Decomposed with compatibility)**: NFD + decomposes compatibility characters
   - Example: ﬁ (ligature) → f + i
   - Use case: Strict equivalence checking

4. **NFKC (Composed with compatibility)**: NFKD + recomposes
   - Example: ﬁ (ligature) → f + i (then recomposed if applicable)
   - Use case: Maximum compatibility; information loss acceptable

### NFC: The Process

NFC (the focus of this report) performs three logical steps:

1. **Decomposition (NFD)**: Break down precomposed characters into base + combining marks using canonical decomposition mappings from `UnicodeData.txt`
2. **Reordering**: Sort combining marks by their canonical combining class to ensure consistent order
3. **Recomposition**: Recombine base characters with their combining marks back into precomposed forms, **except** where explicitly forbidden by `CompositionExclusions.txt`

The key insight: **NFC is NOT simply "recompose everything."** Some characters are intentionally excluded from recomposition via the composition exclusion list. This ensures data integrity and prevents ambiguous mappings.

---

## Section 2: UnicodeData.txt Format & Parsing

### File Overview

`UnicodeData.txt` is the master Unicode character database. It contains 40,575 lines (in this version), with one line per Unicode character. Each line is semicolon-delimited with 15 fields.

### Field-by-Field Breakdown

Consider this real example from the file:

```
00E9;LATIN SMALL LETTER E WITH ACUTE;Ll;0;L;0065 0301;;;;N;LATIN SMALL LETTER E ACUTE;;00C9;;00C9
```

Broken down by field (0-indexed):

| Field | Name | Example | Purpose |
|-------|------|---------|---------|
| 0 | **Code Point** | `00E9` | Hexadecimal Unicode codepoint |
| 1 | **Character Name** | `LATIN SMALL LETTER E WITH ACUTE` | Official Unicode name |
| 2 | **General Category** | `Ll` | `Ll` = lowercase letter; `Lu` = uppercase; `Mn` = nonspacing mark, etc. |
| 3 | **Canonical Combining Class** | `0` | Integer 0-254; determines reordering priority (0 = not a combining mark) |
| 4 | **Bidi Class** | `L` | Bidirectional text behavior (`L`=left-to-right, `R`=right-to-left, etc.) |
| 5 | **Decomposition Mapping** | `0065 0301` | **CRITICAL FOR NFC**: Space-separated codepoints this character decomposes to |
| 6 | **Numeric Value (Decimal)** | (empty) | Numeric value if applicable (digit characters) |
| 7 | **Numeric Value (Digit)** | (empty) | Digit value if applicable |
| 8 | **Numeric Value (Numeric)** | (empty) | General numeric value |
| 9 | **Bidi Mirrored** | `N` | `Y` if character has a mirrored pair (parentheses, etc.) |
| 10 | **Unicode 1.0 Name** | `LATIN SMALL LETTER E ACUTE` | Legacy name from Unicode 1.0 |
| 11 | **ISO Comment** | (empty) | ISO standard reference if applicable |
| 12 | **Simple Uppercase Mapping** | `00C9` | Codepoint this character uppercases to (é → É) |
| 13 | **Simple Lowercase Mapping** | (empty) | Codepoint this character lowercases to |
| 14 | **Simple Titlecase Mapping** | `00C9` | Codepoint for titlecase (usually same as uppercase) |

### Decomposition Mapping: The NFC Lifeline

Field 5 is **essential for NFC**. It tells us how to decompose a character.

**Format**:
- Empty field = character doesn't decompose (is atomic)
- Single space-separated list = canonical decomposition (used for NFC/NFD)
- `<tag> ` prefix = compatibility decomposition (only used for NFKC/NFKD, skipped by NFC)

**Real Examples**:
```
00C9;LATIN CAPITAL LETTER E WITH ACUTE;Lu;0;L;0045 0301;;;;N;...
     ↑ É decomposes into 0045 (E) + 0301 (combining acute)

00E9;LATIN SMALL LETTER E WITH ACUTE;Ll;0;L;0065 0301;;;;N;...
     ↑ é decomposes into 0065 (e) + 0301 (combining acute)

0065;LATIN SMALL LETTER E;Ll;0;L;;;;;N;...
     ↑ e (base letter) has NO decomposition; it's atomic

0301;COMBINING ACUTE ACCENT;Mn;230;NSM;;;;;N;...
     ↑ Combining acute has NO decomposition; it's atomic
     ↑ Field 3 = 230: the canonical combining class (high priority reordering)
```

### Parsing UnicodeData.txt: Pseudo-Code

```
Algorithm: BuildDecompositionMap(unicodedata_file)

decomposition_map := empty dictionary
combining_class_map := empty dictionary

FOR EACH line IN unicodedata_file:
    IF line starts with '#':
        SKIP comment
    
    fields := line.split(';')
    
    codepoint := parse_hex(fields[0])
    combining_class := parse_int(fields[3])
    decomposition_field := fields[5]
    
    // Store combining class (needed for reordering step)
    combining_class_map[codepoint] := combining_class
    
    // Parse decomposition mapping
    IF decomposition_field is not empty:
        IF decomposition_field starts with '<':
            // Compatibility decomposition (has tag like <compat>)
            // Extract tag and codepoints after tag
            tag := extract_tag(decomposition_field)
            codepoints := extract_codepoints_after_tag(decomposition_field)
            
            // NFC ignores compatibility decompositions
            IF tag == '' (canonical, no tag):
                decomposition_map[codepoint] := parse_codepoints(codepoints)
        ELSE:
            // Canonical decomposition (no tag)
            codepoints := decomposition_field.split(' ')
            decomposition_map[codepoint] := parse_codepoints(codepoints)
    
RETURN (decomposition_map, combining_class_map)
```

### Implementation Considerations

When implementing the parser:

1. **Codepoints are hexadecimal strings**: `00E9` must be parsed as integer `0xE9` = `233`
2. **Multiple codepoints**: Some characters decompose to 2+ codepoints; store as an array/list
3. **Empty fields**: Use empty strings or special markers to distinguish "no decomposition" from "compatibility only"
4. **Combining class**: Essential for the reordering step; cache this separately for fast lookup
5. **Skip compatibility decompositions** for NFC (they start with a tag like `<compat>`)

---

## Section 3: CompositionExclusions.txt Format & Purpose

### Why Composition Exclusions Exist?

After NFD decomposition and reordering, we want to recompose characters. However, **not every character should be recomposed**. Some decompositions are excluded to prevent:

1. **Canonical overloading**: Multiple codepoints mapping to the same composed form
2. **Ambiguity**: Characters with multiple valid decompositions
3. **Script-specific rules**: Certain scripts (e.g., Devanagari) have historical reasons to exclude recomposition
4. **Singleton issues**: When a single character decomposes into a sequence, recomposing could create ambiguity

Example of the problem:
```
If we recompose ALL decompositions:
  ï (U+00EF) + combining diaeresis → ???
  
Without exclusions, we might lose the original character's identity.
This is why some characters are forbidden from recomposing.
```

### File Structure & Examples

`CompositionExclusions.txt` is simple: a list of codepoints (in hex) that should NOT be recomposed, with categorized comments.

Real examples from the file:

```
# ================================================
# (1) Script Specifics
#
# This list of characters cannot be derived from the UnicodeData.txt file.
#
# Included are the following subcategories:
# - Many precomposed characters using a nukta diacritic in the Devanagari,
#   Bengali, Assamese, Oriya, Tamil, Telugu, Kannada, and Malayalam scripts.

0958    #  DEVANAGARI LETTER QA
0959    #  DEVANAGARI LETTER KHHA
095A    #  DEVANAGARI LETTER GHHA
...
09DC    #  BENGALI LETTER RRA
09DD    #  BENGALI LETTER RHA
09DF    #  BENGALI LETTER YYA
...
```

### The Four Categories of Exclusions

#### 1. Script Specifics (Singleton Decompositions)

**What**: Characters that decompose to a single codepoint, used in scripts like Devanagari, Bengali, Gurmukhi, etc.

**Why exclude**: Recomposing a single codepoint back to the same character would be pointless. More importantly, these characters often represent base forms with diacritics (nuktas) and have script-specific rules.

**Examples**:
- `0958` (DEVANAGARI LETTER QA): Decomposes to another Devanagari character; recomposing could violate script rules
- `09DC` (BENGALI LETTER RRA): Historical character; should not be normalized away

#### 2. Singleton Decompositions

**What**: A single character mapping to a sequence of characters.

**Why exclude**: If character A decomposes to character B alone, and we recompose it back to A, we gain nothing and lose information about the original form. Recomposing could mask the original identity.

**Example from UnicodeData**:
```
Some characters have decompositions to single characters
that don't compose back to meaningful forms.
```

#### 3. Non-Starter Decompositions

**What**: Decompositions where the first codepoint is a combining mark (category Mn, Ms, Me).

**Why exclude**: NFC recomposition algorithm assumes the first element of a decomposition is a base character, not a combining mark. Allowing these would break the algorithm's invariant.

**Would look like**: A character that decomposes to `<combining_mark> <base>` instead of `<base> <combining_mark>`

#### 4. Canonical Equivalents

**What**: Characters that are canonically equivalent to other precomposed characters; multiple ways to reach the same character.

**Why exclude**: To prevent ambiguous recomposition. If multiple decomposition paths lead to the same character, we must exclude some to maintain a consistent canonical form.

**Example**:
```
Character A + combining mark → Character B
Character C + combining mark → Character B

If we recompose all of them, we'd lose track of which
was the "original" A, C, or their decomposed forms.
```

### Parsing CompositionExclusions.txt: Pseudo-Code

```
Algorithm: BuildCompositionExclusionSet(exclusions_file)

exclusion_set := empty set
exclusion_categories := empty dictionary

FOR EACH line IN exclusions_file:
    line := line.strip()
    
    IF line is empty OR line starts with '#':
        // Track category headers for logging/debugging
        IF line starts with '# (1)' OR '# (2)' OR '# (3)' OR '# (4)':
            current_category := line
        SKIP comment or empty line
    ELSE:
        // Extract codepoint (hex string before optional comment)
        parts := line.split('#')
        codepoint_str := parts[0].strip()
        
        codepoint := parse_hex(codepoint_str)
        
        exclusion_set.add(codepoint)
        exclusion_categories[codepoint] := current_category
        
        // Optional: store the comment for debugging
        IF len(parts) > 1:
            comment := parts[1].strip()
            // Store comment if useful

RETURN exclusion_set
```

### Real Data: What Gets Excluded?

From the file, we see **~115+ codepoints** excluded:
- Devanagari script: `0958-095F`, `0961-0962`
- Bengali script: `09DC-09DD`, `09DF`
- Gurmukhi script: `0A33`, `0A36`, `0A59-0A5B`, `0A5E`
- Oriya script: `0B5C-0B5D`, `0B71`
- Tamil script: `0B94`, `0BCA-0BCC`
- ... and many others

The pattern is clear: **These are script-specific characters with diacritic-like behavior** that shouldn't be decomposed/recomposed under normal NFC rules.

---

## Section 4: The NFC Algorithm (Deep Dive)

### Three-Step Process

NFC normalization operates in three distinct phases:

#### Step 1: Canonical Decomposition (NFD)

**Goal**: Break every precomposed character into its base + combining marks.

**Pseudo-Code**:

```
Algorithm: DecomposeCanonical(text, decomposition_map)

result := empty list of codepoints

FOR EACH codepoint IN text:
    IF codepoint IN decomposition_map:
        // Character has a decomposition mapping
        decomposed := decomposition_map[codepoint]
        
        FOR EACH decomposed_codepoint IN decomposed:
            RECURSIVELY add decomposed_codepoint to result
            // Recursive call in case decomposed char also decomposes
    ELSE:
        // Character is atomic; keep as-is
        APPEND codepoint to result

RETURN result
```

**Why recursive?** Some characters decompose to other precomposable characters.

**Example**:
```
Input: é (U+00E9)
decomposition_map[00E9] = [0065, 0301]
Result after decomposition: [0065 (e), 0301 (combining acute)]

Input: more complex character X
decomposition_map[X] = [Y, Z]
decomposition_map[Y] = [A, B]
Result: [A, B, Z] (fully decomposed)
```

#### Step 2: Canonical Reordering

**Goal**: Sort combining marks by canonical combining class for consistent order.

**The Combining Class**: Each combining mark has a canonical combining class (0-254).
- **0** = base character (not a combining mark)
- **1-230** = various combining marks, sorted by combining class
- Higher numbers = reorder later

**Algorithm**:

```
Algorithm: ReorderByComposingClass(decomposed_list, combining_class_map)

result := empty list

i := 0
WHILE i < length(decomposed_list):
    // Find the next base character (combining class 0)
    base_and_marks := [decomposed_list[i]]
    
    WHILE i + 1 < length(decomposed_list) AND 
          combining_class_map[decomposed_list[i+1]] > 0:
        // Gather all consecutive combining marks
        base_and_marks.append(decomposed_list[i+1])
        i := i + 1
    
    // Sort combining marks by combining class
    marks := base_and_marks[1:]  // Everything after base
    marks.sort(by combining_class)
    
    // Add base + sorted marks to result
    result.append(base_and_marks[0])
    FOR EACH mark IN marks:
        result.append(mark)
    
    i := i + 1

RETURN result
```

**Example**:
```
Input after decomposition: [e, combining_acute(230), combining_diaeresis(230)]

Both combining marks have class 230, so relative order is stable.
But if one had class 220 and another 230, reorder:
  [e, combining(220), combining(230)]
```

#### Step 3: Recomposition (With Exclusions)

**Goal**: Recombine base characters with combining marks, but **skip if in exclusion list**.

**Algorithm**:

```
Algorithm: Recompose(reordered_list, decomposition_map, exclusion_set)

composition_map := invert(decomposition_map)  // [base, mark] → composed
result := empty list

i := 0
WHILE i < length(reordered_list):
    codepoint := reordered_list[i]
    
    IF combining_class_map[codepoint] == 0:  // Base character
        // Try to compose with following combining marks
        last_composed := codepoint
        
        j := i + 1
        WHILE j < length(reordered_list) AND 
              combining_class_map[reordered_list[j]] > 0:
            
            next_mark := reordered_list[j]
            
            // Try to compose base + mark
            composed := lookup_composition(composition_map, last_composed, next_mark)
            
            IF composed != NULL AND composed NOT IN exclusion_set:
                // Recompose!
                last_composed := composed
                j := j + 1  // Skip this mark; it's consumed
            ELSE:
                // Cannot or should not compose; mark is kept separate
                BREAK
        
        // Add the (possibly recomposed) base to result
        APPEND last_composed to result
        
        // Add any remaining combining marks that didn't compose
        WHILE i + 1 < j:
            i := i + 1
            IF i < j:
                APPEND reordered_list[i] to result
        
        i := j
    ELSE:
        // Stray combining mark (shouldn't happen in well-formed text)
        APPEND codepoint to result
        i := i + 1

RETURN result
```

**Key Points**:
- Build composition_map by inverting decomposition_map
- For each base, try to compose with following marks (in order)
- If composition exists **AND** is not excluded, recompose
- Stop when you can't compose with the next mark
- Keep non-composable marks as-is

### Full NFC Example: Walking Through "é + combining grave"

Let's normalize the sequence: **é** (U+00E9) + **combining grave accent** (U+0300)

**Input**: `[00E9, 0300]`

**Step 1: Decompose**

```
00E9 decomposes to [0065, 0301]
0300 has no decomposition

After decomposition: [0065, 0301, 0300]
```

**Step 2: Reorder**

```
combining_class[0065] = 0    (base)
combining_class[0301] = 230  (combining acute)
combining_class[0300] = 230  (combining grave)

Both marks have class 230. Stable sort maintains order:
After reordering: [0065, 0301, 0300]
```

**Step 3: Recompose**

```
i = 0: Process 0065 (base letter 'e')
  j = 1: Try compose(0065, 0301)
    composition_map[0065, 0301] = 00E9 ✓
    00E9 NOT in exclusion_set ✓
    Recompose to 00E9
  
  j = 2: Try compose(00E9, 0300)
    composition_map[00E9, 0300] = NULL ✗
    Cannot compose; break
  
  Add 00E9 to result
  Add remaining mark 0300 to result

Output: [00E9, 0300]
```

**Final Result**: é + combining grave (not further normalized)

Why not further composed? Because no precomposed form exists for "e with acute and grave" in Unicode. The algorithm respects the Unicode database's composition boundaries.

---

## Section 5: Implementation Considerations

### Data Structures for Performance

For a real implementation, you need efficient lookups:

```
decomposition_map: HashMap<codepoint, List<codepoint>>
    Purpose: Fast O(1) lookup of decompositions
    Build from: UnicodeData.txt field 5
    
composition_map: HashMap<(codepoint, codepoint), codepoint>
    Purpose: Fast O(1) lookup of compositions
    Build from: Invert decomposition_map
    (base + mark) → composed
    
combining_class_map: HashMap<codepoint, int>
    Purpose: Fast O(1) lookup of combining class
    Build from: UnicodeData.txt field 3
    
exclusion_set: HashSet<codepoint>
    Purpose: O(1) membership test
    Build from: CompositionExclusions.txt
```

**Memory**: Expect ~100-200 KB for these maps in a typical implementation.

### Caching & Optimization

1. **Cache decomposition results**: If normalizing many strings, memoize decomposition→recomposition pipelines
2. **Lazy initialization**: Don't load all Unicode data if you only need ASCII (though most real apps need full Unicode)
3. **Incremental normalization**: For editors, normalize only changed text, not the entire document
4. **Use precomputed tables**: Many Unicode implementations ship with binary-compiled versions of these maps

### Edge Cases & Gotchas

1. **Hangul Syllables (Korean)**
   - Hangul has its own special decomposition/recomposition rules
   - Not covered in CompositionExclusions.txt; handled separately
   - Most implementations special-case Hangul

2. **Singleton Characters in Decomposition**
   - Some characters decompose to a single codepoint (e.g., A + ring above → Å)
   - After decomposition: [0041, 030A]
   - Recomposition: [00C5] (Å)
   - These are NOT singletons in the exclusion sense; they're normal compositions

3. **Multiple Combining Marks**
   - After decomposing é + combining grave: [0065, 0301, 0300]
   - Recomposition tries to compose 0065 + 0301 first (succeeds)
   - Then tries to compose 00E9 + 0300 (fails; no such character)
   - Result: 00E9 + 0300 (one composed, one not)

4. **Stability & Idempotence**
   - Applying NFC twice should give the same result: NFC(NFC(x)) == NFC(x)
   - This is guaranteed if algorithms are implemented correctly
   - Test this rigorously

5. **Canonical Ordering Matters**
   - If you skip reordering (Step 2), you might fail to recompose later
   - Example: marks in wrong order prevent composition
   - Always include reordering

### Testing Recommendations

Before deploying an NFC implementation, test:

1. **Basic Decomposition-Recomposition**
   ```
   Input:  "café"
   NFC:    "café" (precomposed)
   NFD:    "cafe´" (decomposed)
   NFC(NFD(x)) should equal NFC(x)
   ```

2. **Combining Mark Reordering**
   ```
   Input with out-of-order marks
   Verify: Marks reorder to correct combining class order
   Verify: Recomposition succeeds after reordering
   ```

3. **Exclusion Enforcement**
   ```
   For each codepoint in CompositionExclusions.txt:
     - Verify it does NOT recompose
     - Verify decomposition still works
   ```

4. **Idempotence**
   ```
   FOR EACH test_string IN test_set:
     nfc1 := NFC(test_string)
     nfc2 := NFC(nfc1)
     assert nfc1 == nfc2
   ```

5. **Unicode Standard Conformance**
   ```
   Use official Unicode test cases from:
   https://www.unicode.org/Public/UCD/latest/ucd/NormalizationTest.txt
   ```

6. **Performance Benchmarks**
   ```
   Typical ASCII: Should normalize ~1-10 MB/sec
   Mixed Unicode: Should normalize ~100-500 KB/sec
   Benchmark against reference implementations
   ```

### Common Implementation Mistakes

1. **Forgetting canonical reordering**: Marks in wrong order prevent composition
2. **Ignoring exclusions**: Recomposing excluded characters breaks conformance
3. **Not handling compatibility decompositions**: NFC should skip them (not NFKC)
4. **Building composition map incorrectly**: Inversion must preserve all mappings
5. **Not handling empty/missing decompositions**: Fields can be empty; must distinguish from decomposable
6. **Case sensitivity in hex parsing**: `00e9` vs `00E9` should be equivalent
7. **Recursive decomposition bugs**: If a decomposed character also decomposes, must handle fully

### Performance Analysis

For a string of length n with average combining marks m:

- **Decomposition**: O(n × d) where d = max decomposition depth (usually ≤ 3)
- **Reordering**: O(n × m × log(m)) for sorting marks
- **Recomposition**: O(n × m × 2) for composition lookups

**Overall**: O(n × m × log(m)) in practice, often dominated by I/O if reading large files.

---

## Conclusion

NFC normalization is surprisingly complex, but breaks into manageable steps:

1. **Parse UnicodeData.txt** to extract decomposition mappings and combining classes
2. **Parse CompositionExclusions.txt** to build the exclusion set
3. **Implement** the three-step NFC algorithm: decompose, reorder, recompose
4. **Test thoroughly** against Unicode standard conformance tests

The files provided in this directory (`UnicodeData.txt` and `CompositionExclusions.txt`) contain all the data needed. The challenge is in building efficient data structures and handling edge cases correctly.

For developers implementing this: Start with ASCII (simple), then extend to Latin extended (test decomposition), then add full Unicode support. Use the pseudo-code above as a reference, and test every step with the Unicode conformance test suite.

---

**References & Further Reading**

- Unicode Standard Annex #15 (Normalization): https://www.unicode.org/reports/tr15/
- UCD File Format: https://www.unicode.org/reports/tr44/
- Unicode Normalization Test Cases: https://www.unicode.org/Public/UCD/latest/ucd/NormalizationTest.txt
- Hangul Normalization Special Rules: https://www.unicode.org/reports/tr15/#Hangul

---

**Word Count**: ~2,800 words | **Last Updated**: 2026-03-09
