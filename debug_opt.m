clear all

%penalities:
%====================
%frequency deviations
p_qt = 1;
p_qw = 1;

%virtual damping
p_rd = 0.08;

%virtual inertia
p_rm = 0.008;
%--------------------

%Bus fault size / probability
%==============================
bus1f = 0.05;
bus2f = 1.0;
bus3f = 0.08;
bus4f = 1.0;
bus5f = 0.5;
number_busses = 5;
II = eye(number_busses);      % size n-by-n matrix
%II = [bus1f,0.0,0.0,0.0,0.0; 0.0,bus2f,0.0,0.0,0.0; 0.0,0.0,bus3f,0.0,0.0; 0.0,0.0,0.0,bus4f,0.0; 0.0,0.0,0.0,0.0,bus5f]*100;
%------------------------------

%objective function
fun =@(mg) find_Pm(p_qt,p_qw,p_rd,p_rm,II, mg(1),mg(2),mg(3),mg(4),mg(5),mg(6),mg(7),mg(8),mg(9),mg(10));


% bounds
lb = [zeros(1,5), 0.0, 0.0, 0.0, 0.0, 0.0];
ub = [130e3,130e3,130e3,130e3,130e3, 25e6,25e6,25e6,25e6,25e6];

% linear inequality
A = [zeros(1,5), ones(1,5); ones(1,5), zeros(1,5)];
b = [50e6;300e3];

% linear equality
Aeq = []; 
beq = []; 

% initial value
m0 = [1e3,10e3,10e3,1e3,1e3, 10e6,10e6,10e6,10e6,10e6];

% optimization
options = optimoptions(@fmincon,'MaxFunctionEvaluations',50000);
[v, fval] = fmincon(fun,m0,A,b,Aeq,beq,lb,ub,[],options);


scaleM = 1e6;
unitsM = " MWs^2";
scaleV = 1e3;
unitsV = " kWs";
% Display output
S = "VD1 = " + num2str(round(v(1)/(scaleV),3)) + unitsV +  "   VM1 = " + num2str(round(v(1 + 5)/(scaleM),3)) + unitsM + newline + "VD2 = " + num2str(round(v(2)/scaleV,3)) + unitsV + "   VM2 = " + num2str(round(v(2 + 5)/(scaleM),3)) + unitsM + newline + "VD3 = " + num2str(round(v(3)/scaleV,3)) + unitsV  + "   VM3 = " + num2str(round(v(3 + 5)/(scaleM),3)) + unitsM + newline + "VD4 = " + num2str(round(v(4)/scaleV,3)) + unitsV  + "   VM4 = " + num2str(round(v(4 + 5)/(scaleM),3))+ unitsM + newline + "VD5 = " + num2str(round(v(5)/scaleV,3)) + unitsV  + "   VM5 = " + num2str(round(v(5 + 5)/(scaleM),3)) + unitsM + newline;

disp(S + newline + "Hardy2 = " + num2str(fval));













function Hardy2 = find_Pm(p_qt,p_qw,p_rd,p_rm,II_defined, vd1,vd2,vd3,vd4,vd5,vm1,vm2,vm3,vm4,vm5)

subMatrix = [2 3 4 5];

v1 = 345000;%15000
v2 = 345000;
v3 = 345000;%15000
v4 = 345000;
v5 = 345000;

v11 = v1*v1;
v15 = v1*v5;
v22 = v2*v2;
v24 = v2*v4;
v25 = v2*v5;
v33 = v3*v3;
v34 = v3*v4;
v44 = v4*v4;
v45 = v4*v5;
v55 = v5*v5;

% [actual]
%---------------------------------
Zbase2 = 1190.25;
ZbaseT = 1190.25;

B21 = 0; % x tends to infinity

B31 = 0; % x tends to infinity
B32 = 0; % x tends to infinity

B41 = 0; % x tends to infinity
B42 = v24/(0.1*Zbase2);
B43 = v34/( (0.01*ZbaseT) );

B51 = v15/( (0.02*ZbaseT) );
B52 = v25/(0.05*Zbase2);
B53 = 0; % x tends to infinity
B54 = v45/(0.025*Zbase2);
%=================================


B_og = [-1*(B21+B31+B41+B51)    B21                    B31                   B41                    B51;
     B21                        -1*(B21+B32+B42+B52)   B32                   B42                    B52;
     B31                        B32                    -1*(B31+B32+B43+B53)  B43                    B53;
     B41                        B42                    B43                   -1*(B41+B42+B43+B54)   B54;
     B51                        B52                    B53                   B54                    -1*(B51+B52+B53+B54)];


B_sub = B_og(subMatrix, subMatrix);


P_inject = [-8e8;
            4.4e8;
              0;
              0];
          
 angle_rad = (-B_sub)\P_inject;
angle_deg = (angle_rad/pi)*180;  %remove (;) to display


%========================================================================================================================
%Interconnected synchronous machines

% Inertia (SyncM)

%Converts the powerfactory parameter H to M for calculation
%H1 = 4;
%S1rated = 800e6; %[VA]
%m1 = (2*H1*S1rated)/314.159;                % Bus1 (primary bus)
m1 = 20.3718e6;

%H2 = 1e-0;
%S2rated = 800e6; %[VA]                      % Bus2
%m2 = (2*H2*S2rated)/314.159;
m2 = 0.01e6;

%H3 = 4;
%S3rated = 800e6; %[VA]
%m3 = (2*H3*S3rated)/314.159;                % Bus3 (second generator)
m3 = 25.4648e6;

%H4 = 1e-0;
%S4rated = 800e6; %[VA]                      % Bus4
%m4 = (2*H4*S4rated)/314.159;
m4 = 0.01e6;

%H5 = 1e-0;
%S5rated = 800e6; %[VA]                      % Bus5
%m5 = (2*H5*S5rated)/314.159;
m5 = 0.01e6;


M =  [m1  0  0  0  0;
      0   m2 0  0  0;
      0   0  m3 0  0;
      0   0  0  m4 0;
      0   0  0  0  m5];

M_sub = M(subMatrix, subMatrix);


% Damping (SyncM)
d1 = 3e6;              % Bus 1 (primary bus)
d2 = 0.01e6;               % Bus 2
d3 = 4e6;              % Bus 3
d4 = 0.01e6;               % Bus 4
d5 = 0.02e6;               % Bus 5 -- 0.01

D = [d1  0  0  0  0;
      0  d2 0  0  0;
      0  0  d3 0  0;
      0  0  0  d4 0;
      0  0  0  0  d5];
  
%D_sub = D(subMatrix, subMatrix);  



 A0 = [zeros(size(M,1)) eye(size(M,1)); (-M)\(-B_og) (-M)\D];
 %[PDT, A0D] = eig(A0);

  
B0 = [zeros(size(M)); inv(M)];

C0 = [zeros(size(M,1)), eye(size(M,1))];

% (similarity transforms)
[P, A_delta] = eig(A0);             % size 2n-by-2n
A_delta = A_delta([1 2 3 4 5 6 7 8 9], [1 2 3 4 5 6 7 8 9]);

B_delta = P\B0;  %inv(P)*B0;
%B_delta = B_delta_og([1 2 3 4 5 6 7 8 10], [1 2 3 4 5]);
B_delta = B_delta([1 2 3 4 5 6 7 8 9], [1 2 3 4 5]);


C_delta = C0*P;
C_delta = C_delta([1 2 3 4 5], [1 2 3 4 5 6 7 8 9]);


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
Tg2 = 2;  %2;    %[p.u.]
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
Tg4 = 2;  %2;    %[p.u.]
Rg4 = inf;  %0.05; %[p.u.]

Kg4 = 1/Rg4;

Ag4 = -1/Tg4;
Bg4 = 0;
Cg4 = -Kg4;

%(Governor 5)
Tg5 = 2;  %2;    %[p.u.]
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
      
   

Bg = [Bg1    0    0    0    0;
       0    Bg2   0    0    0;
       0     0   Bg3   0    0;
       0     0    0   Bg4   0;
       0     0    0    0   Bg5];
   

  
Cg = [Cg1    0    0    0    0;
       0    Cg2   0    0    0;
       0     0   Cg3   0    0;
       0     0    0   Cg4   0;
       0     0    0    0   Cg5];
 


%============================================================================================
% Virtual inertia as feedback control loop
% //selection of state vaiables x_i = (wi_tilde_i, wi_tilde_dot_i)
% // yout = [pdi;
%            pmi]

T1 = 0.01;  % PD-control causality time constant
T2 = 0.03;   % Time constant to model the PLL

% Virtual inertia per bus i

A_tilde_i = [(T1+T2)/(-1*T1*T2), -1/(T1*T2);1,0];
     
B_tilde_i = [1/(T1*T2);
                0];
            
            
% Aggregated A_tilde_i                                    % size 2n-by-2n
A_tilde = [A_tilde_i       zeros(2)        zeros(2)        zeros(2)        zeros(2);
            zeros(2)       A_tilde_i       zeros(2)        zeros(2)        zeros(2);
            zeros(2)       zeros(2)       A_tilde_i        zeros(2)        zeros(2);
            zeros(2)       zeros(2)        zeros(2)       A_tilde_i        zeros(2);
            zeros(2)       zeros(2)        zeros(2)        zeros(2)       A_tilde_i];


        
% Aggregated B_tilde_i
B_tilde = [B_tilde_i       zeros(2,1)   zeros(2,1)   zeros(2,1)   zeros(2,1);
           zeros(2,1)      B_tilde_i    zeros(2,1)   zeros(2,1)   zeros(2,1);
           zeros(2,1)      zeros(2,1)   B_tilde_i    zeros(2,1)   zeros(2,1);
           zeros(2,1)      zeros(2,1)   zeros(2,1)   B_tilde_i    zeros(2,1);
           zeros(2,1)      zeros(2,1)   zeros(2,1)   zeros(2,1)   B_tilde_i];
       

        
%==============================================================================================
% Closed-loop system

II = II_defined;

IIg = eye(size(M,1), size(M,1));

B_delta_g = B_delta*IIg'*Cg;
%B_delta_g = B_delta_g([1 2 3 4 5 6 7 8 9], [1 2 3 4 5]);


% IGNORE!!! manual adjusting to get rid of the top half of matrix which just contains
% zeros ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
%C_delta_sub = C_delta([5 6 7 8], [1 2 3 4 5 6 7 8]);
%B_g_delta = Bg_sub*IIg*C_delta_sub; 
B_g_delta = Bg*IIg*C_delta;
B_g_delta = B_g_delta([1 2 3 4 5], [1 2 3 4 5 6 7 8 9]);

%B_tilde_C_delta = B_tilde*C_delta;
%B_tilde_C_delta = B_tilde_C_delta([1 2 3 4 5 6 7 8 9 10], [1 2 3 4 5 6 7 8 9]);

%A = [A_delta                        B_delta_g                                       zeros( 2*size(M_sub,1) , size(A_tilde_sub,2));
%     B_g_delta                      Ag_sub                                          zeros( 1*size(M_sub,1) , size(A_tilde_sub,2));
%     B_tilde_sub*C_delta            zeros( size(A_tilde_sub,1) , size(Ag_sub,2))    A_tilde_sub];
 
A = [A_delta                        B_delta_g                                   zeros(size(A_delta,1),size(A_tilde,2));
     B_g_delta                      Ag                                          zeros(size(Ag,1),size(A_tilde,2));
     B_tilde*C_delta                zeros(size(A_tilde,1),size(Ag,2))           A_tilde];


 
Bmv = kron(B_delta, [1 1]); 
 

B = [Bmv;
     zeros(size(A,1)-size(Bmv,1), size(Bmv,2))];
 
 
G =[B_delta*II;
    zeros(size(A,1)-size(B_delta*II,1), size(B_delta*II,2))];


% Feedback gain matrix
% (from H to M)

d1 = vd1;%10   %20   %10e6 
d2 = vd2;%0.1   %10e3   %1
d3 = vd3;%10   %20   %10e5
d4 = vd4;%0.1   %100   %1
d5 = vd5;%0.1   %100e6   %100e6

                         %20.3718,25.4648            %5,4
m1 = vm1;%10   %10    %2       %10e6 sss                     %All (1e6)               % m1 <= 0.00001     //on generator
m2 = vm2;%0.1   %920       %0.1                        %0.1                     % m2 <= 0.0006
m3 = vm3;%20   %3       %100e6                      %All (1e6)               % m3 <= 0.00004  //on generator
m4 = vm4;%0.1   %11        %0.1                        %0.1                     % m4 <= 0.00002
m5 = vm5;%0.1   %All       %All                        %All(100e6)              % m5 is stable for all local values

%m1 = (2*vm1*S1rated)/314.159;            % [10 0.0001]      //on generator
%m2 = (2*vm2*S2rated)/314.159;            % [10 0.0001]
%m3 = (2*vm3*S3rated)/314.159;            % [0.001 0.0001]  //on generator
%m4 = (2*vm4*S4rated)/314.159;            % [0.0001]
%m5 = (2*vm5*S5rated)/314.159;            % [0.01 0.0001]  // 0.01 gee H2 = 17.2823

C_tilde = [m1   0   0   0   0   0   0   0   0   0;
            0   d1  0   0   0   0   0   0   0   0;
            0   0   m2  0   0   0   0   0   0   0;
            0   0   0   d2  0   0   0   0   0   0;
            0   0   0   0   m3  0   0   0   0   0;
            0   0   0   0   0   d3  0   0   0   0;
            0   0   0   0   0   0   m4  0   0   0;
            0   0   0   0   0   0   0   d4  0   0;
            0   0   0   0   0   0   0   0   m5  0;
            0   0   0   0   0   0   0   0   0   d5;];
        
%K_tilde_proef = [zeros(3*size(M_sub,1))                       zeros(3*size(M_sub,1), 2*size(M_sub,1));
%                 zeros(2*size(M_sub,1), 3*size(M_sub,1))      C_tilde];
 
% size is 2(n-1)-by-5(n-1)
K_tilde = [zeros(2*size(M,1),(2*size(M,1))-1) zeros(2*size(M,1),size(M,1)) C_tilde]; 
 
  
Q = [p_qt*eye(size(M,1)); p_qw*eye(size(M,1))];

 
% size 2(n-1)-by-2(n-1)
% sqrtm(Q)*C_delta
C = [Q*C_delta                              zeros(size(Q*C_delta,1),size(M,1))     zeros(size(Q*C_delta,1),2*size(M,1));
     zeros(size(M,1),size(Q*C_delta,2))     zeros(size(M,1))                       zeros(size(M,1),2*size(M,1));
     zeros(2*size(M,1),size(Q*C_delta,2))   zeros(2*size(M,1),size(M,1))           zeros(2*size(M,1))]; 

%-------------------------------------------------
% Tune parameters to penalizes the control effort
rm = p_rm;  %0.0083;
rd = p_rd;    %0.08;
%--------------------------------------------------

R = kron(eye(size(M,1)),[rd 0; 0 rm]);

% size 5(n-1)-by-2(n-1)
F = [zeros(2*size(M_sub,1), 2*size(M,1));
     zeros(size(M,1), 2*size(M,1));
     sqrtm(R)]; 


%==================================================================================================
Acl = A - B*K_tilde;
QVM = C'*C + K_tilde'*F'*F*K_tilde;
 
eig(Acl); %remove (;) to display

% solving lyapunov ( Pk = lyap(Acl', QVM) )
Pk = lyap(Acl', QVM);


format short e
Hardy2 =  round( sqrt( (trace(G'*Pk*G)) ) ,5) 
format long

end