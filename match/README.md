
### Folders

* `strict`:
  Contains diffraction and structure data of **strictly matched** RRUFF entries. Identified structures share the same elements and lattice constants.

* `relaxed`:
  Contains diffraction and structure data of **relaxed matched** RRUFF entries. Identified structures may have different elements but share the same lattice constants.

* `MP_data`:
  Contains diffraction and structure data of MP entries.
  **If an unstable connection interrupts the download of this folder, a ZIP archive is also available on [HuggingFace](https://huggingface.co/datasets/caobin/PyXplore/resolve/main/MP_data.zip?download=true).**

Note: All matches are subject to certain tolerances, as no two crystals are perfectly identical. The reference structure derived from **XQueryer** serves as the input for further structure determination during the refinement step.





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
