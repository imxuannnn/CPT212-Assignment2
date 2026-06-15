"""

This module provides two implementations of the Boyer-Moore algorithm:
  Method 1 - Bad Character Heuristic only
  Method 2 - Bad Character + Good Suffix Heuristic (full algorithm)

Both methods are tested on identical inputs so that their results and
comparison counts can be contrasted.

HOW BOYER-MOORE WORKS (Overview):
  Unlike naive search which checks every position left-to-right, Boyer-Moore
  compares characters RIGHT-TO-LEFT within each alignment. When a mismatch
  happens, it uses clever rules to skip ahead — often jumping over large
  chunks of text without checking them at all.

  Two "skip rules" (heuristics) are used:
    1. Bad Character  — looks at the mismatched text character and asks:
       "Where does this character last appear in my pattern?"
    2. Good Suffix    — looks at the part that already matched and asks:
       "Does this matched part appear elsewhere in my pattern?"

  The algorithm picks whichever rule gives the BIGGER skip.

Time Complexity:
  - Best / Average : O(n / m)  — skips large chunks, very efficient
  - Worst case     : O(n * m)  — rare, happens with highly repetitive text
  - Preprocessing  : O(m + 256) — one-time cost to build lookup tables
  (n = length of text, m = length of pattern)

"""

# There are 256 possible characters in the extended ASCII table (0 to 255).
# We create a table with one slot per character.
NO_OF_CHARS = 256

# Preprocessing: Bad Character Heuristic

def bad_char_heuristic(pattern):
    """
    Build the bad character table.

    WHAT THIS TABLE ANSWERS:
      "For any character, what is the LAST position where it appears in the pattern?"

    WHY THIS IS USEFUL:
      When a mismatch occurs, we look at the text character that caused it.
      - If that character EXISTS in the pattern → slide the pattern so that
        its last occurrence lines up with the mismatched position.
      - If that character does NOT exist in the pattern → we get -1,
        which means we can skip the entire pattern past that position.

    EXAMPLE:
      pattern = "ABCD"
        table['A'] = 0   (A last appears at position 0)
        table['B'] = 1   (B last appears at position 1)
        table['C'] = 2   (C last appears at position 2)
        table['D'] = 3   (D last appears at position 3)
        table[any other character] = -1   (not in pattern → big skip)

    Time:  O(m + 256)  where m = length of pattern
    Space: O(256)      one slot per possible ASCII character
    """
    # Start with -1 for all 256 characters (meaning "not found in pattern")
    table = [-1] * NO_OF_CHARS

    # Scan through the pattern left to right.
    # If a character appears more than once, the later position naturally
    # overwrites the earlier one — so we automatically keep the LAST (rightmost) position.
    for i in range(len(pattern)):
        table[ord(pattern[i])] = i      # ord() converts a character to its ASCII number

    return table


# Preprocessing: Good Suffix Heuristic

def good_suffix_preprocess(pattern):
    """
    Build the good suffix shift table.

    WHAT THIS TABLE ANSWERS:
      "After some characters at the end of the pattern have matched (the 'good suffix'),
       but then a mismatch occurs — how far can we safely slide the pattern forward?"

    TWO CASES ARE HANDLED:

      Case 1 — Strong Good Suffix:
        The matched suffix appears AGAIN somewhere else inside the pattern,
        and the character just BEFORE that other occurrence is DIFFERENT from
        the mismatched character. We slide the pattern to align that copy.

        Example: pattern = "ABCABC"
          If we matched "ABC" at the end and hit a mismatch, we can slide
          the pattern to align the first "ABC" with the text.

      Case 2 — Matching Prefix (fallback):
        The matched suffix does NOT appear elsewhere, but the BEGINNING
        of the pattern matches the END of the good suffix.
        We slide the pattern to align that matching prefix.

        Example: pattern = "ABAABA"
          If we matched "ABA" at the end but can't find another "ABA",
          we check: does the pattern start with "A" or "ABA"? If yes,
          align that prefix with the text.

    HOW IT WORKS INTERNALLY:
      - border[i] tracks where the suffix starting at position i also appears
        elsewhere in the pattern (used for the internal computation).
      - shift[j] stores the final answer: how far to slide the pattern when
        a mismatch occurs at position j.

    Time:  O(m)
    Space: O(m)
    """
    m = len(pattern)
    shift = [0] * (m + 1)      # shift[j]: how far to slide when mismatch at position j-1
    border = [0] * (m + 1)     # border[i]: helper array tracking suffix borders

    # ---- Case 1: Strong Good Suffix ----
    # We scan the pattern from right to left, looking for places where
    # a suffix of the pattern also appears elsewhere in the pattern.
    i = m           # Start at the end of the pattern
    j = m + 1       # One past the end (starting reference point)
    border[i] = j

    while i > 0:
        # If characters DON'T match, it means the suffix starting at j
        # differs from the one starting at i — so we found a useful shift.
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i      # Only record the first (closest) match
            j = border[j]             # Follow the chain to try the next border

        # Characters matched — extend the border one step to the left
        i -= 1
        j -= 1
        border[i] = j

    # ---- Case 2: Matching Prefix (fallback) ----
    # Fill in any shift positions that Case 1 left as 0 (unset).
    # For these, we use the longest prefix of the pattern that is also a suffix.
    j = border[0]       # Start with the widest border of the entire pattern
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j    # Use this border length as a safe fallback shift

        # When we reach the boundary of the current prefix,
        # move to the next shorter matching prefix
        if i == j:
            j = border[j]

    return shift


# Search Method 1: Bad Character Heuristic Only

def bm_bad_char_only(text, pattern):
    """
    Boyer-Moore search using ONLY the Bad Character Heuristic.

    This is the simpler version. It only uses one skip rule:
    when a mismatch occurs, it looks at the mismatched text character
    and shifts the pattern based on where that character last appears
    in the pattern.

    This method is easier to understand but may not skip as efficiently
    as the full algorithm when the pattern has repeated suffixes.

    Returns:
      matches     - list of starting positions where pattern was found
      comparisons - total number of character comparisons made

    Preprocessing : O(m + 256)
    Search        : O(n*m) worst case, O(n/m) best/average case
    """
    n = len(text)
    m = len(pattern)

    # Nothing to search if pattern is empty or longer than the text
    if m == 0 or m > n:
        return [], 0

    # Build the bad character lookup table
    bc = bad_char_heuristic(pattern)

    matches = []        # Will store all positions where pattern is found
    comparisons = 0     # Counter to track how many character comparisons we make
    s = 0               # Current alignment: pattern starts at text[s]

    # Slide the pattern across the text from left to right
    while s <= n - m:
        j = m - 1       # Start comparing from the LAST (rightmost) character

        # Move leftward as long as characters match
        while j >= 0 and pattern[j] == text[s + j]:
            comparisons += 1    # Count each successful comparison
            j -= 1

        if j < 0:
            # j dropped below 0 → ALL characters matched → found the pattern!
            matches.append(s)
            s += 1              # Shift by 1 to find any overlapping matches
        else:
            comparisons += 1    # Count the mismatch comparison too

            # Use the bad character rule to calculate the shift.
            # bc[ord(text[s + j])] gives the last position of the mismatched
            # text character in the pattern. Subtracting from j gives the shift.
            # If the character isn't in the pattern at all (bc = -1), we get a big skip.
            bc_shift = j - bc[ord(text[s + j])]

            # Always shift at least 1 to avoid getting stuck
            s += max(1, bc_shift)

    return matches, comparisons


# Search Method 2: Bad Character + Good Suffix (Full BM)

def bm_full(text, pattern):
    """
    Full Boyer-Moore search using BOTH heuristics:
      1. Bad Character Heuristic
      2. Good Suffix Heuristic

    This is the complete algorithm. When a mismatch occurs, it calculates
    two possible shifts (one from each heuristic) and picks the LARGER one.
    This is what makes Boyer-Moore so efficient — it always skips as far
    ahead as it safely can.

    Returns:
      matches     - list of starting positions where pattern was found
      comparisons - total number of character comparisons made

    Preprocessing : O(m + 256)
    Search        : O(n) worst case (with good suffix), O(n/m) best/average case
    """
    n = len(text)
    m = len(pattern)

    # Nothing to search if pattern is empty or longer than the text
    if m == 0 or m > n:
        return [], 0

    # Build BOTH lookup tables during preprocessing
    bc = bad_char_heuristic(pattern)    # Bad character table
    gs = good_suffix_preprocess(pattern) # Good suffix shift table

    matches = []        # Will store all positions where pattern is found
    comparisons = 0     # Counter for character comparisons
    s = 0               # Current alignment position

    # Slide the pattern across the text from left to right
    while s <= n - m:
        j = m - 1       # Start comparing from the rightmost character

        # Move leftward as long as characters match
        while j >= 0 and pattern[j] == text[s + j]:
            comparisons += 1
            j -= 1

        if j < 0:
            # ALL characters matched → pattern found at position s!
            matches.append(s)

            # Use the good suffix table for the shift after a full match.
            # gs[0] gives the proper shift when the entire pattern matched.
            s += gs[0]
        else:
            comparisons += 1    # Count the mismatch comparison

            # Calculate BOTH shifts and pick the bigger one:

            # Bad character shift: align the mismatched text character
            # with its last occurrence in the pattern
            bc_shift = j - bc[ord(text[s + j])]

            # Good suffix shift: use the precomputed table based on
            # how many characters already matched (the "good suffix")
            gs_shift = gs[j + 1]

            # Take the MAXIMUM — this is the key to Boyer-Moore's speed!
            s += max(bc_shift, gs_shift)

    return matches, comparisons


# Driver - Demonstration & Comparison

if __name__ == "__main__":
    print("=" * 66)
    print(" Boyer-Moore Algorithm - Dual-Method Demonstration")
    print("=" * 66)

    # Test 1: repeating pattern with multiple matches
    text1    = "ABCABCABDABABCABC"
    pattern1 = "ABCABC"
    r1a, c1a = bm_bad_char_only(text1, pattern1)
    r1b, c1b = bm_full(text1, pattern1)
    print(f"\nTest 1  text = \"{text1}\"  pattern = \"{pattern1}\"")
    print(f"  BCH-only   -> matches {r1a}, comparisons = {c1a}")
    print(f"  Full BM    -> matches {r1b}, comparisons = {c1b}")

    # Test 2: single match in an English sentence 
    text2    = "HERE IS A SIMPLE EXAMPLE"
    pattern2 = "EXAMPLE"
    r2a, c2a = bm_bad_char_only(text2, pattern2)
    r2b, c2b = bm_full(text2, pattern2)
    print(f"\nTest 2  text = \"{text2}\"  pattern = \"{pattern2}\"")
    print(f"  BCH-only   -> matches {r2a}, comparisons = {c2a}")
    print(f"  Full BM    -> matches {r2b}, comparisons = {c2b}")

    # Test 3: DNA sequence (small alphabet → more repetitions)
    text3    = "GCATCGCAGAGAGTATACAGTACG"
    pattern3 = "GCAGAGAG"
    r3a, c3a = bm_bad_char_only(text3, pattern3)
    r3b, c3b = bm_full(text3, pattern3)
    print(f"\nTest 3  text = \"{text3}\"  pattern = \"{pattern3}\"")
    print(f"  BCH-only   -> matches {r3a}, comparisons = {c3a}")
    print(f"  Full BM    -> matches {r3b}, comparisons = {c3b}")

    # Test 4: pattern does not exist in text
    text4    = ""
    pattern4 = "ZEBRA"
    r4a, c4a = bm_bad_char_only(text4, pattern4)
    r4b, c4b = bm_full(text4, pattern4)
    print(f"\nTest 4  text = \"{text4}\"  pattern = \"{pattern4}\"")
    print(f"  BCH-only   -> matches {r4a}, comparisons = {c4a}")
    print(f"  Full BM    -> matches {r4b}, comparisons = {c4b}")

    # Test 5: short example for step-by-step tracing 
    text5    = "AECDABCD"
    pattern5 = "ABCD"
    r5a, c5a = bm_bad_char_only(text5, pattern5)
    r5b, c5b = bm_full(text5, pattern5)
    print(f"\nTest 5 (trace)  text = \"{text5}\"  pattern = \"{pattern5}\"")
    print(f"  BCH-only   -> matches {r5a}, comparisons = {c5a}")
    print(f"  Full BM    -> matches {r5b}, comparisons = {c5b}")

    print("\n" + "=" * 66)
    print(" All tests completed.")
    print("=" * 66)
