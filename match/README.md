
## RRUFF–MP ID Matching

This folder contains code and data for matching mineral entries from the RRUFF database with corresponding materials in the Materials Project (MP) database. The main matching script is located in [`./main.py`](./main.py).

### Overview

We first converted the RRUFF database into a structured database file (`.db`) by matching RRUFF CIF IDs with its PDF IDs. For each entry, we saved its crystal structure, diffraction pattern, space group, and other metadata. Due to licensing restrictions from the RRUFF project, this converted database **cannot be shared publicly**. However, if you are using it for research purposes, feel free to contact me directly at **[bcao686@connect.hkust-gz.edu.cn](mailto:bcao686@connect.hkust-gz.edu.cn)**, and I’d be happy to share it personally.

### Matching Criteria

A two-fold matching strategy is applied:

1. **Strict Matching**:
   We require that the matched MP entry has:

   * The **same elemental composition** as the RRUFF entry.
   * **Lattice constants within 5%** deviation.
     These pairs are labeled as `strict` in the file [`matched_pairs.txt`](./matched_pairs.txt).

2. **Relaxed Matching**:
   We allow:

   * **Lattice constants within 1%** deviation, even if exact elemental matches are not present.
     These pairs are labeled as `relaxed` in the same file.

The second matching type is motivated by the fact that RRUFF is a relatively small database, and many structures not be present in the MP dataset. Therefore, this relaxed matching allows us to capture similar crystal structures that induce **similar XRD peak distributions**, even if they are not identical in composition. This aligns with the *in-library identification* concept discussed in the supplementary materials of our paper.

### Files

* `matched_pairs.txt`:
  Contains all matched RRUFF and MP ID pairs, e.g.:

  ```
  Matched RRUFFID=R040031 <--> MPID=mp-10851.cif (strict)
  ```

* `matched_dict.pkl`:
  A Python dictionary storing all matched pairs for easier programmatic access.

* `correct_ids.txt`:
  The correct identified rruff data id.

* `main.py`:
  The core matching script that performs the ID matching process and includes detailed logic for both strict and relaxed criteria.
