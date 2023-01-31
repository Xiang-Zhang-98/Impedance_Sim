#include <stdio.h>
#include <math.h>

#define PI 3.1415926535

// Robot parameters
double L[3] = {5, 4, 3};


// Direct inverse of a 3x3 matrix m
void mat3x3_inv(double minv[3][3], const double m[3][3])
{
 double det;
 double invdet;
 det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
 m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
 m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
 invdet = 1.0 / det;
 minv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
 minv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
 minv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
 minv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
 minv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
 minv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
 minv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
 minv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
 minv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;
}
/* Explicit 3x3 matrix * vector. b = A x where A is 3x3 */
void mat3x3_vec_mult(double b[3], const double A[3][3], const double x[3])
{
 const int N = 3;
 int idx, jdx;
 for( idx = 0; idx < N; idx++ )
 {
 b[idx] = 0.0;
 for( jdx = 0; jdx < N; jdx++)
 {
 b[idx] += A[idx][jdx] * x[jdx] ;
 }
 }
}

// Compute Jacobian matrix based on the joint configuration
void Jacobian(double J[3][3], const double q[3])
{
    J[0][0] = - L[0] * sin(q[0]) - L[1] * sin(q[0] + q[1]) - L[2] * sin(q[0] + q[1] + q[2]);
    J[0][1] = - L[1] * sin(q[0] + q[1]) - L[2] * sin(q[0] + q[1] + q[2]);
    J[0][2] = - L[2] * sin(q[0] + q[1] + q[2]);
    J[1][0] = L[0] * cos(q[0]) + L[1] * cos(q[0] + q[1]) + L[2] * cos(q[0] + q[1] + q[2]);
    J[1][1] = L[1] * cos(q[0] + q[1]) + L[2] * cos(q[0] + q[1] + q[2]);
    J[1][2] = L[2] * cos(q[0] + q[1] + q[2]);
    J[2][0] = 1;
    J[2][1] = 1;
    J[2][2] = 1;
}

// Inverse kinematics of 3 link planner robot
void Inverse_Kinematics(double q[3], const double eff[3])
{
    // compute wrist position
    double xw, yw, alpha, beta, gamma, r;
    xw = eff[0] - L[2] * cos(eff[2]);
    yw = eff[1] - L[2] * sin(eff[2]);
    r = sqrt(xw * xw + yw * yw);
    // check feasibility with triangle inequality
    if (r > L[0] + L[1] || r < fabs(L[0] - L[1]))
    {
        // no IK solution
        q[0] = NAN;
        q[1] = NAN;
        q[2] = NAN;
    }
    else
    {
        gamma = acos((r * r + L[0] * L[0] - L[1] * L[1]) / (2 * L[0] * r));
        beta = acos((L[0] * L[0] + L[1] * L[1] - r * r) / (2 * L[0] * L[1]));
        alpha = atan2(yw, xw);
        q[0] = alpha - gamma;
        q[1] = PI - beta;
        q[2] = eff[2] - q[0] - q[1];
    }
}

// Compute determinate
double det(const double a[3][3])
{
    double determinant = a[0][0] * ((a[1][1]*a[2][2]) - (a[2][1]*a[1][2])) -a[0][1] * (a[1][0] * a[2][2] - a[2][0] * a[1][2])
                  + a[0][2] * (a[1][0] * a[2][1] - a[2][0] * a[1][1]);
    return determinant;
}

// pick and place planning
int pick_and_place(const double eff_init[3], const double eff_final[3], const double t_d, const int N)
{
    // Compute distance between the init and final point
    double delta[3] = {eff_final[0]-eff_init[0], eff_final[1]-eff_init[1], eff_final[2]-eff_init[2]};
    // Compute the cartesian space velocity
    double ve[3] = {delta[0]/t_d, delta[1]/t_d, delta[2]/t_d};

    double waypoint[3], q[3], q_dot[3], J[3][3], J_inv[3][3];
    double path[N][6];
    int infeasible =  0;

    // If N < 1, then number of waypoints is not enough
    if(N < 1)
    {
        printf("Invalid number of intermediate points!\n");
        return 1;
    }
    // If t_d <= 0, then desired time is invalid
    if(t_d <= 0)
    {
        printf("Invalid desired time!\n");
        return 1;
    }

    printf("q1           q2           q3           dq1           dq2           dq3\n");
    int idx, jdx;
    for( idx = 0; idx < N; idx++ )
    {
        // Interpolate to get waypoint
        waypoint[0] = eff_init[0] + (idx+1) * delta[0] / (N + 1);
        waypoint[1] = eff_init[1] + (idx+1) * delta[1] / (N + 1);
        waypoint[2] = eff_init[2] + (idx+1) * delta[2] / (N + 1);

        // Solve IK to get joint configuration
        Inverse_Kinematics(q, waypoint);
        // Checking whether IK solution is feasible
        if(isnan(q[0]) || isnan(q[1]) || isnan(q[2]))
        {
            printf("This waypoint is out of the workspace and no IK solution exists!\n");
            infeasible = 1;
            continue;
        }
        // Get Jacobian
        Jacobian(J, q);

        // Check singularity
        if(det(J)>1e-3)
        {
            // q_dot = J_inv @ ve
            mat3x3_inv(J_inv, J);
            mat3x3_vec_mult(q_dot, J_inv, ve);
        }
        else
        {
            printf("This waypoint is too close to the singularity!\n");
            infeasible = 1;
            continue;
        }

        // save to the path and print
        for( jdx = 0; jdx < 3; jdx++ )
        {
            path[idx][jdx] = q[jdx];
            printf("%6lf     ", q[jdx]);
        }

        for( jdx = 0; jdx < 3; jdx++ )
        {
            path[idx][jdx+3] = q_dot[jdx];
            printf("%6lf     ", q_dot[jdx]);
        }
        printf("\n");
    }
    return infeasible;
}

int main(void)
{
    printf("Please select mode:\n0: use pre-defined init and final point ([3,3,0] to [5,5,0], t_d = 2.0, N = 10)\nothers: use user input\n");
    int i, N;
    double eff_init[3], eff_final[3], t_d;
    int mode;
    scanf("%d",&mode);
    switch(mode) {
      case 0 :
        // use default input
        N = 10;
        eff_init[0]=3; eff_init[1]=3; eff_init[2]=0;
        eff_final[0]=5;eff_final[1]=5;eff_final[2]=0;
        t_d=2.0;
        break;
      default :
        // get function input
        printf("Input the initial end-effector pose\n");
        for( i = 0; i < 3; i++ )
        {
            scanf("%lf",&eff_init[i]);
        }
        printf("Input the final end-effector pose\n");
        for( i = 0; i < 3; i++ )
        {
            scanf("%lf",&eff_final[i]);
        }
        printf("Input the desired time \n");
        scanf("%lf",&t_d);
        printf("Input the number of intermediate points \n");
        scanf("%d",&N);
   }
    int result;
    result = pick_and_place(eff_init, eff_final, t_d, N);
    return result;
}
