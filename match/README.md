
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
