rocks3d-rr3: 21999 training pairs, 510 test/val pairs. Test & val are the same.
rocks3d-rr4: 13780 training pairs, 250 test/val pairs. Test & val are the same.
rocks3d-rr3-rr4-mix: 12299 training pairs, 250 test/val pairs. Test & val are the same.

rocks3d: 
- Each rock model has 7 partial LiDAR sets and the model is permuted by 16 times, so each model has 112 partial-gt pairs
- In total 46+36=82 RR3 and RR4 rock models, so the dataset has 82x112=9184 partial-gt pairs.
- partial clouds is 2048 points, gt clouds is 16384 points.
- 9000 training pairs, 184 test/val pairs.
- To shuffle the dataset, we add uuid at the beginning of the filename and the rock information at the end of the filename.