clear all;
% Note: their is manual adjusting down at ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

%subMatrix = [2 3 4 5]; % vector discribes the sub-matrix row and colomn selection




% calculate H2 norm
%H2 = trace(B_delta'*Pk*B_delta)
%calcPmB_delta = @(x1,x2) find_Pm(x1,x2);

%objective function
%fun = @(x1,x2) trace( (calcPmB_delta(2))'*calcPmB_delta(1)*(calcPmB_delta(2)) );
fun =@(mg) find_Pm(mg(1),mg(2),mg(3),mg(4),mg(5),mg(6),mg(7),mg(8),mg(9),mg(10));

% bounds
lb = [zeros(1,5), 0.0001, 0.0001, 0.0001, 0.0001, 0.0001];
ub = [ones(1,5), 15*ones(1,5)];
%ub = [ones(1,5), 15, 0.0001, 15, 0.0001, 0.0001];

% linear inequality
A = [];
b = [];

% linear equality
Aeq = [zeros(1,5), ones(1,5)]; %[1,1]
beq = 7;

% initial value
m0 = [0.1*ones(1,5), 3,0,4,0,0]; %[11,4]

% optimization
[v, fval] = fmincon(fun,m0,A,b,Aeq,beq,lb,ub);

S = "VD1 = " + num2str(v(1)) +  "   VM1 = " + num2str(v(1 + 5)) + newline + "VD2 = " + num2str(v(2)) + "   VM2 = " + num2str(v(2 + 5)) + newline + "VD3 = " + num2str(v(3)) + "   VM3 = " + num2str(v(3 + 5)) + newline + "VD4 = " + num2str(v(4)) + "   VM4 = " + num2str(v(4 + 5)) + newline + "VD5 = " + num2str(v(5)) + "   VM5 = " + num2str(v(5 + 5)) + newline;

disp(S + newline + "Hardy2 = " + num2str(fval));
 

 %############ Open-loop system #####################
 %Gs_open = ss(A_delta,B_delta,C_delta,0);
 %###################################################
 
 %H2 = round( normh2(Gs_open,0.0001), 5)
 
 function Hardy2 = find_Pm(vd1,vd2,vd3,vd4,vd5,vm1,vm2,vm3,vm4,vm5)

 subMatrix = [2 3 4 5]; % vector discribes the sub-matrix row and colomn selection

 %=========================================================================================================================
%   "DC"-Powerflow bus suscepatance matrix of the grid
%   Power injection vector
%   Node-bus voltage angles


% [p.u.]
%---------------------------------
%B21 = 0; % x tends to infinity

%B31 = 0; % x tends to infinity
%B32 = 0; % x tends to infinity

%B41 = 0; % x tends to infinity
%B42 = 1/0.1;
%B43 = 1/( 0.08/8 );

%B51 = 1/( 0.16/8 );
%B52 = 1/0.05;
%B53 = 0; % x tends to infinity
%B54 = 1/0.025;
%=================================




% [actual]
%---------------------------------
Zbase2 = 1190.25;
ZbaseT = 1190.25;

B21 = 0; % x tends to infinity

B31 = 0; % x tends to infinity
B32 = 0; % x tends to infinity

B41 = 0; % x tends to infinity
B42 = 1/(0.1*Zbase2);
B43 = 1/( (0.01*ZbaseT) );

B51 = 1/( (0.02*ZbaseT) );
B52 = 1/(0.05*Zbase2);
B53 = 0; % x tends to infinity
B54 = 1/(0.025*Zbase2);
%=================================


% NOT ANYMORE [in p.u.]
B_og = [-1*(B21+B31+B41+B51)   B21                    B31                   B41                    B51;
     B21                    -1*(B21+B32+B42+B52)   B32                   B42                    B52;
     B31                    B32                    -1*(B31+B32+B43+B53)   B43                    B53;
     B41                    B42                    B43                   -1*(B41+B42+B43+B54)   B54;
     B51                    B52                    B53                   B54                    -1*(B51+B52+B53+B54)];

 
%B = [B_og(:,2) B_og(:,3) B_og(:,4) B_og(:,5)];
%B = [B(2,:); B(3,:); B(4,:); B(5,:)];
%B_sub = B_og(subMatrix, subMatrix);


% [in p.u. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!]
%P_inject = [-8;
%            4.4;
%              0;
%              0];
 
%angle_rad = (-((345*1000)^2)*B_sub)\P_inject;
%angle_rad = -1*inv(B_sub)*P_inject;
%angle_deg = (angle_rad/pi)*180;


%========================================================================================================================
%Interconnected synchronous machines

% [should be in p.u. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!]
% Inertia (SyncM)
%m0 = [7,8];

H1 = 4;
H2 =
H3
H4
H5


m1 = 4; %7.2;                 % Bus1 (primary bus) 
m2 = 0.01;                 % Bus2
m3 = 4; %7.8;                 % Bus3 (second generator)
m4 = 0.01;                 % Bus4
m5 = 0.01;                 % Bus5

% optimal inertia for two identical rated sync gens (m1 = 7.2, m3 = 7.8)

M =  [m1  0  0  0  0;
      0   m2 0  0  0;
      0   0  m3 0  0;
      0   0  0  m4 0;
      0   0  0  0  m5];

M_sub = M(subMatrix, subMatrix);



% [should be in p.u. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!]
% Damping (SyncM)
d1 = 0.01;              % Bus 1 (primary bus)
d2 = 0.01;               % Bus 2
d3 = 0.01;              % Bus 3
d4 = 0.01;               % Bus 4
d5 = 0.01;               % Bus 5

D = [d1  0  0  0  0;
      0  d2 0  0  0;
      0  0  d3 0  0;
      0  0  0  d4 0;
      0  0  0  0  d5];
  
%D_sub = D(subMatrix, subMatrix);  


%A0 = [zeros(size(M_sub,1)) eye(size(M_sub,1)); (-1*M_sub)\B_sub (-1*M_sub)\D_sub];
% experiment for A0
 A0 = [zeros(size(M,1)) eye(size(M,1)); (-M)\(-B_og) (-M)\D];
 %[PDT, A0D] = eig(A0);

  
B0 = [zeros(size(M)); inv(M)];

% // OLD // C0 = [zeros(1,size(M_sub,1)) ones(1,size(M_sub,1))];
%C0 = [zeros(size(M,1)) zeros(size(M,1)); zeros(size(M,1)) eye(size(M,1))];
C0 = [zeros(size(M,1)), eye(size(M,1))];

% (similarity transforms)
[P, A_delta] = eig(A0);             % size 2n-by-2n
A_delta = A_delta([1 2 3 4 5 6 7 8], [1 2 3 4 5 6 7 8]);

B_delta_og = P\B0;  %inv(P)*B0;
B_delta = B_delta_og([1 2 3 4 5 6 7 8], [1 2 3 4]);

% Output -----------------------------
% Bm_delta = B_delta;

C_delta = C0*P;
% (OLD) C_delta = C_delta([1 2 3 4 5 6 7 8], [1 2 3 4 5 6 7 8]);
C_delta = C_delta([1 2 3 4], [1 2 3 4 5 6 7 8]);

% ****************state vector size is 2*(n-1)-by-1

%========================================================================================================================
%   Governors and primary droop control

%(Governor 1)
Tg1 = 2;    %[p.u.]
Rg1 = 0.05; %[p.u.]

Kg1 = 1/Rg1;

Ag1 = -1/Tg1;
Bg1 = 1/Tg1;
Cg1 = -Kg1;

%(Governor 2)
Tg2 = Tg1;  %2;    %[p.u.]
Rg2 = inf;    %0.05; %[p.u.]

Kg2 = 1/Rg2;

Ag2 = -1/Tg2;
Bg2 = 0;
Cg2 = -Kg2;

%(Governor 3)
Tg3 = 2;  %2;    %[p.u.]
Rg3 = 0.05;  %0.05; %[p.u.]

Kg3 = 1/Rg3;

Ag3 = -1/Tg3;
Bg3 = 1/Tg3;
Cg3 = -Kg3;

%(Governor 4)
Tg4 = Tg1;  %2;    %[p.u.]
Rg4 = inf;  %0.05; %[p.u.]

Kg4 = 1/Rg4;

Ag4 = -1/Tg4;
Bg4 = 0;
Cg4 = -Kg4;

%(Governor 5)
Tg5 = Tg1;  %2;    %[p.u.]
Rg5 = inf;  %0.05; %[p.u.]

Kg5 = 1/Rg5;

Ag5 = -1/Tg5;
Bg5 = 0;
Cg5 = -Kg5;

% (Aggregated model)
Ag = [Ag1  0    0   0   0;              % size n-by-n
       0  Ag2   0   0   0;
       0   0   Ag3  0   0;
       0   0    0  Ag4  0;
       0   0    0   0  Ag5];
   
%Ag_sub = Ag(subMatrix, subMatrix);   
   

Bg = [Bg1    0    0    0    0;
       0    Bg2   0    0    0;
       0     0   Bg3   0    0;
       0     0    0   Bg4   0;
       0     0    0    0   Bg5];
   
%Bg_sub = Bg(subMatrix, subMatrix);

  
Cg = [Cg1    0    0    0    0;
       0    Cg2   0    0    0;
       0     0   Cg3   0    0;
       0     0    0   Cg4   0;
       0     0    0    0   Cg5];
 
%Cg_sub = Cg(subMatrix, subMatrix);

% ****************state vector size is (n-1)-by-1

%============================================================================================
% Virtual inertia as feedback control loop
% //selection of state vaiables x_i = (wi_tilde_i, wi_tilde_dot_i)
% // yout = [pdi;
%            pmi]

T1 = 0.01;  % PD-control causality time constant
T2 = 0.05;   % Time constant to model the PLL

% Virtual inertia per bus i
A_tilde_i = [0                  1;
            -1/(T1*T2)  (T1+T2)/(-1*T1*T2)];
     
B_tilde_i = [1/(T1*T2);
                0];
            
            
% Aggregated A_tilde_i                                    % size 2n-by-2n
A_tilde = [A_tilde_i       zeros(2)        zeros(2)        zeros(2)        zeros(2);
            zeros(2)       A_tilde_i       zeros(2)        zeros(2)        zeros(2);
            zeros(2)       zeros(2)       A_tilde_i        zeros(2)        zeros(2);
            zeros(2)       zeros(2)        zeros(2)       A_tilde_i        zeros(2);
            zeros(2)       zeros(2)        zeros(2)        zeros(2)       A_tilde_i];

%A_tilde_sub = A_tilde(subMatrix, subMatrix);
 %A_tilde_sub = kron( eye(size(M_sub,1)), A_tilde_i );
        
% Aggregated B_tilde_i
B_tilde = [B_tilde_i       zeros(2,1)   zeros(2,1)   zeros(2,1)   zeros(2,1);
           zeros(2,1)      B_tilde_i    zeros(2,1)   zeros(2,1)   zeros(2,1);
           zeros(2,1)      zeros(2,1)   B_tilde_i    zeros(2,1)   zeros(2,1);
           zeros(2,1)      zeros(2,1)   zeros(2,1)   B_tilde_i    zeros(2,1);
           zeros(2,1)      zeros(2,1)   zeros(2,1)   zeros(2,1)   B_tilde_i];
       
%B_tilde_sub = B_tilde(subMatrix, subMatrix);
 %B_tilde_sub = kron( eye(size(M_sub,1)), B_tilde_i);
   B_tilde_sub = kron( eye(size(M,1), size(M_sub,1)), B_tilde_i);

% ****************state vector size is 2*(n-1)-by-1
        
%==============================================================================================
% Closed-loop system
II = eye(size(M_sub, 1));      % size n-by-n matrix
%IIg = eye(size(M_sub, 1));     % size n-by-1 vector
IIg = eye(size(M,1),size(M_sub,1));

B_delta_g = B_delta*IIg'*Cg;


% IGNORE!!! manual adjusting to get rid of the top half of matrix which just contains
% zeros ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
%C_delta_sub = C_delta([5 6 7 8], [1 2 3 4 5 6 7 8]);
%B_g_delta = Bg_sub*IIg*C_delta_sub; 
B_g_delta = Bg*IIg*C_delta;


%A = [A_delta                        B_delta_g                                       zeros( 2*size(M_sub,1) , size(A_tilde_sub,2));
%     B_g_delta                      Ag_sub                                          zeros( 1*size(M_sub,1) , size(A_tilde_sub,2));
%     B_tilde_sub*C_delta            zeros( size(A_tilde_sub,1) , size(Ag_sub,2))    A_tilde_sub];
 
A = [A_delta                        B_delta_g                                   zeros(size(A_delta,1),size(A_tilde,2));
     B_g_delta                      Ag                                          zeros(size(Ag,1),size(A_tilde,2));
     B_tilde_sub*C_delta            zeros(size(A_tilde,1),size(Ag,2))           A_tilde];


 
Bmv = kron(B_delta_og, [1 1]); 
 

B = [Bmv;
     zeros(size(A,1)-size(Bmv,1), size(Bmv,2))];
 
 
G =[B_delta*II;
    zeros(size(A,1)-size(B_delta*II,1), size(B_delta*II,2))];


% Feedback gain matrix


d1 = vd1; 
d2 = vd2;
d3 = vd3;
d4 = vd4;
d5 = vd5;
m1 = vm1;            % [10 0.0001]      //on generator
m2 = vm2;            % [10 0.0001]
m3 = vm3;            % [0.001 0.0001]  //on generator
m4 = vm4;            % [0.0001]
m5 = vm5;            % [0.01 0.0001]  // 0.01 gee H2 = 17.2823

C_tilde = [d1   0   0   0   0   0   0   0   0   0;
            0   m1  0   0   0   0   0   0   0   0;
            0   0   d2  0   0   0   0   0   0   0;
            0   0   0   m2  0   0   0   0   0   0;
            0   0   0   0   d3  0   0   0   0   0;
            0   0   0   0   0   m3  0   0   0   0;
            0   0   0   0   0   0   d4  0   0   0;
            0   0   0   0   0   0   0   m4  0   0;
            0   0   0   0   0   0   0   0   d5  0;
            0   0   0   0   0   0   0   0   0   m5;];
        
%K_tilde_proef = [zeros(3*size(M_sub,1))                       zeros(3*size(M_sub,1), 2*size(M_sub,1));
%                 zeros(2*size(M_sub,1), 3*size(M_sub,1))      C_tilde];
 
% size is 2(n-1)-by-5(n-1)
K_tilde = [zeros(2*size(M,1),2*size(M_sub,1)) zeros(2*size(M,1),size(M,1)) C_tilde]; 
 
  

% Attention!!! Q is singular and may not have a square root
%Q = [zeros(size(M_sub,1))   zeros(size(M_sub,1));
%     zeros(size(M_sub,1))   eye(size(M_sub,1))];

% Q penalizes the frequency deviations
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Q = [0.1*eye(size(M_sub,1))     zeros(size(M_sub,1));
%     zeros(size(M_sub,1))   100*eye(size(M_sub,1))];
Q = [1*eye(size(M_sub,1)); 100*eye(size(M_sub,1))];

 
% size 2(n-1)-by-2(n-1)
% sqrtm(Q)*C_delta
C = [Q*C_delta                              zeros(size(Q*C_delta,1),size(M,1))     zeros(size(Q*C_delta,1),2*size(M,1));
     zeros(size(M,1),size(Q*C_delta,2))     zeros(size(M,1))                       zeros(size(M,1),2*size(M,1));
     zeros(2*size(M,1),size(Q*C_delta,2))   zeros(2*size(M,1),size(M,1))           zeros(2*size(M,1))]; 

%-------------------------------------------------
% Tune parameters to penalizes the control effort
rm = 0.001;  %0.0083;
rd = 0.01;    %0.08;
%--------------------------------------------------

R = kron(eye(size(M,1)),[rd 0; 0 rm]);

% size 5(n-1)-by-2(n-1)
F = [zeros(2*size(M_sub,1), 2*size(M,1));
     zeros(size(M,1), 2*size(M,1));
     sqrtm(R)]; 


%==================================================================================================
Acl = A + B*K_tilde;
QVM = C'*C + K_tilde'*F'*F*K_tilde;
 %CC = C_delta'*C_delta;

% solving lyapunov ( Pk = lyap(Acl', QVM) )
n = size(QVM,1);
A1 = kron(Acl',eye(n)) + kron(eye(n),Acl');
qvm = QVM(:);
x0 = -A1\qvm;
Pk = reshape(x0,n,n);
%Acl'*Pk + Pk*Acl + QVM

%Output-----------------------------------------------
   %h2 = round( sqrt(abs(trace(G'*Pk*G)) ),10)
Hardy2 = round( sqrt(abs(trace(G'*Pk*G)) ),10);
end