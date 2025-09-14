
## RRUFF–MP ID Matching

This folder contains code and data for matching mineral entries from the RRUFF database with corresponding materials in the Materials Project (MP) database. The main matching script is located in [`./main.py`](./main.py).

### Overview

We first converted the RRUFF database into a structured database file (`.db`) by matching RRUFF CIF IDs with its PDF IDs. For each entry, we saved its crystal structure, diffraction pattern, space group, and other metadata. Due to licensing restrictions from the RRUFF project, this converted database **cannot be shared publicly**. However, if you are using it for research purposes, feel free to contact me directly at **[bcao686@connect.hkust-gz.edu.cn](mailto:bcao686@connect.hkust-gz.edu.cn)**, and I’d be happy to share it personally.

### Folders 

* `strict`:  
  Contains diffraction data and structure data of strictly matched RRUFF entries.

* `relaxed`:  
  Contains diffraction data and structure data of relaxed matched RRUFF entries.

* `MP_data`:  
  Contains diffraction data and structure data of matched MP entries.  
  **If an unstable connection interrupts the download of the entire folder, a ZIP archive is also available on [HuggingFace](https://huggingface.co/datasets/caobin/PyXplore/resolve/main/MP_data.zip?download=true).**



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
