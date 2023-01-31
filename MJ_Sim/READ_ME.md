## README
Answer of Systems Analyst Design Problem from Xiang Zhang

## To run the code

- Compile c with gcc:
`gcc -o hw homework_Xiang_Zhang.c -lm`

- run the program:
`./hw`

## Instructions

- This program computes the robot joint angle and velocity of the intermediate points given the init and final end-effector pose, desired time, number of intermediate points.
- This program ask you to select modein the beginning. Input 0 means using default test case (move from (3,3,0) to (5,5,0) over two seconds and output 10 points.). Input any other number to type in the user defined parameters (init and final point, desired time, number of intermediate points).

## Example output

********************************************************************************
Please select mode:
0: use pre-defined init and final point ([3,3,0] to [5,5,0], t_d = 2.0, N = 10)
others: use user input
0
q1           q2           q3           dq1           dq2           dq3
0.587793     2.451323     -3.039116     -0.280318     -0.264127     0.544446     
0.541249     2.402148     -2.943397     -0.232542     -0.276554     0.509096     
0.502921     2.350833     -2.853754     -0.189898     -0.287733     0.477630     
0.471907     2.297571     -2.769478     -0.152003     -0.298034     0.450037     
0.447390     2.242492     -2.689882     -0.118348     -0.307766     0.426114     
0.428647     2.185675     -2.614322     -0.088391     -0.317186     0.405576     
0.415056     2.127157     -2.542213     -0.061602     -0.326517     0.388119     
0.406085     2.066935     -2.473019     -0.037488     -0.335962     0.373450     
0.401289     2.004970     -2.406259     -0.015602     -0.345712     0.361314     
0.400301     1.941191     -2.341492     0.004461     -0.355958     0.351497
********************************************************************************

