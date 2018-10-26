      MODULE PitzCaCl2
C
C     +++++++
!      PRIVATE
C     +++++++
C     ++++++
      PUBLIC :: CaCl2
C     ++++++
C
C
        CONTAINS
C
C
      Subroutine CaCl2(T, mCaCl2, b0_1, b1_1, C0_1, C1_1,
     >                            b0_2, b1_2, C0_2, C1_2,
     >                            D_MX3,D_MX4,D_MX5,
     >                            D_NX3,D_NX4,D_NX5,
     >                            D_MNX2,D_MNX3,
     >                            mCaCl, mCa, mCl, alpha, osm, lnGamma)
C
      IMPLICIT REAL*8 (A-H,O-Z)
C
C     --------------------------------------------------------------------
C     The value of T must be 2981.5D0 Kelvin. This model is for 25 oC only.
C
C     --------------------------------------------------------------------
C
C     Declare f2py intentions
!f2py intent(in) T, mCaCl2, b0_1, b1_1, C0_1, C1_1, b0_2, b1_2, C0_2, C1_2
!f2py intent(in) D_MX3, D_MX4, D_MX5, D_NX3, D_NX4, D_NX5, D_MNX2, D_MNX3
!f2py intent(out) mCaCl, mCa, mCl, alpha, osm, lnGamma
C
C
      LOGICAL :: FirstCall = .TRUE.
      REAL*8  :: KConst, mCaCl2, mCaCl, mCa, mCl, lnGamma,
     >           lnactCa, lnactCaCl, lnactCl
C
      SAVE FirstCall, DHPara

cc      EXTERNAL FC05
C
C
C
      ALPHA1 = 1.4D0
      OMEGA1 = 1.D0
      ALPHA2 = 1.4D0
      OMEGA2 = 2.5D0
      theta  = 0.D0
      psi    = 0.D0
C
      KConst = EXP(-7.488D0)
C
C
!   ..Archer's value
!     --------------
      IF(FirstCall) DHpara = AFT(T)
C
C
        mCa=ZBRENT(FC05,0.D0,mCaCl2,KCONST,DHpara,mCaCl2,
     >                    b0_1,b1_1,C0_1,C1_1,
     >                    b0_2,b1_2,C0_2,C1_2,
     >                    D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                    D_MNX2,D_MNX3,
     >                    alpha1,omega1,alpha2,omega2,
     >                    theta,psi,
     >                    lnactCa,lnactCaCl,lnactCl)
C
        mCaCl=mCaCl2-mCa
        mCl=2*mCaCl2-mCaCl
C
C
C  ..Calculate stoichiometric mean activity coefficient here:
C
        dum=mCa*mCl**2*EXP(lnactCa)*EXP(lnactCl)**2/(4.D0*mCaCl2**3)
        lnGamma = LOG(dum) / 3.D0

C
C
C  ..Degree of dissociation of CaCl+:
        alpha=mCa/(mCaCl2)
C
C
C  ..Calculate the osmotic coefficient:
        CALL OSMCALC(mCaCl2,mCa,mCaCl,mCl,DHpara,
     >                   b0_1,b1_1,C0_1,C1_1,
     >                   b0_2,b1_2,C0_2,C1_2,
     >                   D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                   D_MNX2,D_MNX3,
     >                   alpha1,omega1,alpha2,omega2,
     >                   theta,psi,OSM)
C
      FirstCall = .FALSE.

C
C     ******
      RETURN
C     ******
C
      END Subroutine
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION AFT(T)
      IMPLICIT REAL*8 (A-H,O-Z)
C
C     function yields molality based DH coefficient for absolute
C     temperature T (K) at 1 atm pressure.
C
C message: includes extrapolations for <0oC and >100oC.
C
      PARAMETER(R=8.3144D0, Tr = 273.15D0, Tr2=373.15D0,
     >          AphiTr = 0.376421485D0,    AphiTr2=0.45988678D0,
     >          AlRTr = 0.600305325D0,     AlTr2=5353.2092D0,
     >          AjR = 1.677818299D0,       AjTr2=60.361346D0,
     >          dAjRdT = 0.1753875763D0,   dAjTr2dT=0.528859D0,
     >          N = 17,
     >          XMIN = 234.15D0, XMAX = 373.15D0)
C
C--------------------------------------------------------------------
C               Tr must be >=234.15K,
C               Aphi = DH constant at Tr,
C               AlRTr = AL / (RTr) at Tr,
C               AjR = AJ / R at Tr,
C               dAjRdT = d (AJ/R) / dT at Tr.
C--------------------------------------------------------------------
C
      REAL*8 AA(0:18),TX(0:18)
C
      DATA AA/0.797256081240D+00,0.573389669896D-01,
     > 0.977632177788D-03, 0.489973732417D-02,-0.313151784342D-02,
     > 0.179145971002D-02,-0.920584241844D-03, 0.443862726879D-03,
     >-0.203661129991D-03, 0.900924147948D-04,-0.388189392385D-04,
     > 0.164245088592D-04,-0.686031972567D-05, 0.283455806377D-05,
     >-0.115641433004D-05, 0.461489672579D-06,-0.177069754948D-06,
     > 0.612464488231D-07,-0.175689013085D-07/
C
      If(T.GE.Tr .AND. T.LE.Tr2) then
C
C for Tr2>T>Tr, polynomial reproduces Archer's values:
C
        X=(2.D0*T-XMAX-XMIN)/(XMAX-XMIN)
        TX(0)=1.D0
        TX(1)=X
        AFT=0.5*AA(0)*TX(0) + AA(1)*TX(1)
        DO 1 I=1,N
          TX(I+1)=2*X*TX(I) - TX(I-1)
          AFT=AFT + AA(I+1)*TX(I+1)
1       CONTINUE
C
      ElseIf(T.GT.Tr2) THEN
C
C     ..extrapolation based on Archer. Added 25 May 1995.
C
        AFT= AphiTr2 + AlTr2/(4*R)*(1.D0/Tr2-1.D0/T)
     >     + AjTr2/(4*R)*(LOG(T/Tr2)+Tr2/T-1.D0)
     >     + dAjTr2dT/(8*R)*(T-Tr2**2/T-2*Tr2*LOG(T/Tr2))
C
      ElseIf(T.LT.Tr) THEN
C
C for T<Tr, we have an empirical extrapolation:
C
        AlTr = AlRTr * R * Tr
        AjTr = AjR * R
        dAJdT = dAjRdT * R
C       ---------------------------------------------------
C     ..!Value of a below is that used for 30 Dec R2 run !!
        a=1.45824246467D0
C       ---------------------------------------------------
        x=(dAJdT - a)/(2.D0*Tr)
C
        AFT=AphiTr + AlTr/(4*R)*(1.D0/Tr-1.D0/T)
     >      + AjTr/(4.D0*R)*(LOG(T/Tr)+Tr/T-1.D0)
     >      + a/(8.D0*R)*(T-Tr**2/T-2.D0*Tr*LOG(T/Tr))
     >      + x/(4.D0*R)*(T**2/6.D0+Tr**2/2.D0-Tr**2*LOG(T/Tr)
     >                    -2.D0/3.D0*Tr**3/T)
C
      Endif
C
      END Function
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION AHFT(T)
      IMPLICIT REAL*8 (A-H,O-Z)
C
C     function yields molality based DH enthalpy coefficient for
C     absolute temperature T (K) from 0 to 100oC.
C
      PARAMETER(R=8.3144D0,
     >          N = 12,
     >          XMIN = 273.15D0, XMAX = 373.15D0)
C
      REAL*8 AA(0:13),TX(0:13)
C
      DATA AA/ 0.2228963593D+1,0.5697907469D+0,0.4309490474D-1,
     >        -0.5342823726D-2,0.4652554050D-2,-0.1666733221D-2,
     >         0.5919640117D-3,-0.1880197707D-3,0.5598171457D-4,
     >        -0.1567632212D-4,0.4174053258D-5,-0.1074161684D-5,
     >         0.2607664247D-6,-0.6546599432D-7/
C
      If(T.LT.XMIN .OR. T.GT.XMAX) THEN
        WRITE(2,'(1X,''T OUT OF RANGE IN AHFT: STOP. T = '',E10.3)') T
        STOP
      ENDIF
C
        X=(2.D0*T-XMAX-XMIN)/(XMAX-XMIN)
        TX(0)=1.D0
        TX(1)=X
        AHFT=0.5*AA(0)*TX(0) + AA(1)*TX(1)
        DO 1 I=1,N
          TX(I+1)=2*X*TX(I) - TX(I-1)
          AHFT=AHFT + AA(I+1)*TX(I+1)
1       CONTINUE
C
        AHFT=AHFT * R*T
      END Function
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION ACFT(T)
      IMPLICIT REAL*8 (A-H,O-Z)
C
C     function yields molality based DH heat capacity coefficient for
C     absolute temperature T (K) from 0 to 100oC.
C
      PARAMETER(R=8.3144D0,
     >          N = 14,
     >          XMIN = 273.15D0, XMAX = 373.15D0)
C
      REAL*8 AA(0:15),TX(0:15)
C
      DATA AA/ 0.9477863416D+01,0.2494847484D+01,-0.1593102539D+00,
     >         0.2518621027D+00,-0.9536865366D-01,0.3934452994D-01,
     >        -0.1387023689D-01,0.4562499313D-02,-0.1393392611D-02,
     >         0.4027195964D-03,-0.1113008130D-03,0.2976477353D-04,
     >        -0.7771494186D-05,0.2005540692D-05,-0.4992370416D-06,
     >         0.1339933280D-06/
C
      If(T.LT.XMIN .OR. T.GT.XMAX) THEN
        WRITE(2,'(1X,''T OUT OF RANGE IN ACFT: STOP. T = '',E10.3)') T
        STOP
      ENDIF
C
        X=(2.D0*T-XMAX-XMIN)/(XMAX-XMIN)
        TX(0)=1.D0
        TX(1)=X
        ACFT=0.5*AA(0)*TX(0) + AA(1)*TX(1)
        DO 1 I=1,N
          TX(I+1)=2*X*TX(I) - TX(I-1)
          ACFT=ACFT + AA(I+1)*TX(I+1)
1       CONTINUE
C
        ACFT=ACFT * R
C
      END Function
C
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C All the routines below should be identical in FIT2.FOR and TEST2.FOR
C They are: HFUNC     EDFUNC    FC05
C           GFUNC     PARASET   ZBRENT
C           GDFUNC    OSMCALC
C           EFUNC     ACTCOEF
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION HFUNC(OMEGA,SQRTIX)
       IMPLICIT REAL*8(A-H,O-Z)
C
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-
C                                                             -
C new h(x) function.                                          -
C                                                             -
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-
C
       X=OMEGA*SQRTIX
       HFUNC=(6.D0 - (6.D0+X*(6.D0+3.D0*X+X**2))*EXP(-X))/X**4
      END Function
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION GFUNC(ALPHA,SQRTIX)
       IMPLICIT REAL*8(A-H,O-Z)
C
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-
C                                                             -
C Function g(x),                                              -
C                                                             -
C Used by func BFUNC                                          -
C                                                             -
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-
C
       DUM=ALPHA*SQRTIX
       GFUNC=2.D0*(1.D0-(1.D0+DUM)*EXP(-DUM))/DUM**2
      END Function
C
C
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION GDFUNC(ALPHA,SQRTIX)
       IMPLICIT REAL*8(A-H,O-Z)
C
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-
C                                                             -
C Function g'(x)                                              -
C                                                             -
C Used by func BDFUNC                                         -
C                                                             -
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-
C
       DUM=ALPHA*SQRTIX
       GDFUNC=-2.D0*(1.D0-(1.D0+DUM+5.D-1*DUM**2)*EXP(-DUM))/DUM**2
      END Function
C
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION EFUNC(ICHARG,JCHARG,A,XION)
       IMPLICIT REAL*8(A-H,O-Z)
       REAL*8 XIJ(3),J0(3)
C
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-
C                                                             -
C Calculates E(theta) (unsymmetrical mixing function)         -
C                                                             -
C Used by func EDFUNC, PFUNC, PPFNC                           -
C                                                             -
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-
C
       DUM=6.D0*A*SQRT(XION)
       XIJ(1)=ICHARG*JCHARG*DUM
       XIJ(2)=ICHARG**2*DUM
       XIJ(3)=JCHARG**2*DUM
       DO 1 I=1,3
         J0(I)=XIJ(I)/(4.D0+4.581D0*XIJ(I)**(-7.238D-1)*EXP(-1.2D-2*
     >         XIJ(I)**5.28D-1))
1      CONTINUE
       EFUNC=ICHARG*JCHARG/(4.D0*XION)*(J0(1)-5.D-1*J0(2)-5.D-1*J0(3))
      END Function
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION EDFUNC(ICHARG,JCHARG,A,XION,ETHETA)
       IMPLICIT REAL*8(A-H,O-Z)
       REAL*8 XIJ(3),J1(3)
C
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
C                                                              -
C Calculates E(theta)' (diff. of unsymmetrical mixing function)-
C                                                              -
C Used by func FFUNC, PPFNC                                    -
C                                                              -
C:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:
C
          DUM=6.D0*A*SQRT(XION)
          XIJ(1)=ICHARG*JCHARG*DUM
          XIJ(2)=ICHARG**2*DUM
          XIJ(3)=JCHARG**2*DUM
          DO 1 I=1,3
            DUM=-1.2D-2*XIJ(I)**5.28D-1
            J1(I)=(4.D0+4.581D0*XIJ(I)**(-7.238D-1)*EXP(DUM)*(1.D0+
     >            7.238D-1+1.2D-2*5.28D-1*XIJ(I)**5.28D-1))/(4.D0+
     >            4.581D0*XIJ(I)**(-7.238D-1)*EXP(DUM))**2
1         CONTINUE
          EDFUNC=ICHARG*JCHARG/(8.D0*XION**2)*(XIJ(1)*J1(1)-5.D-1*
     >           XIJ(2)*J1(2)-5.D-1*XIJ(3)*J1(3))
     >           -ETHETA/XION
      END Function
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE PARASET(T,COEFF,pb0_1,pb1_1,pC0_1,pC1_1,
     >                           pb0_2,pb1_2,pC0_2,pC1_2,
     >                           pD_MX3,pD_MX4,pD_MX5,
     >                           pD_NX3,pD_NX4,pD_NX5,
     >                           pD_MNX2,pD_MNX3,
     >                           alpha1,omega1,alpha2,omega2,
     >                           theta,psi)
      IMPLICIT REAL*8(A-H,O-Z)
C
      PARAMETER(NCOEF=100, TR=298.15D0)
C
      REAL*8 COEFF(NCOEF)
C
      D1=T-TR
C
C....................................................................
C
C
      ALPHA1 = COEFF(41)
      OMEGA1 = COEFF(42)
      ALPHA2 = COEFF(43)
      OMEGA2 = COEFF(44)
      theta  = COEFF(45)
      psi    = COEFF(46)
C
C     ..calculate parameter values at temperature T(K)
        pb0_1=COEFF(1) + D1*(COEFF(2)*1.D-1 +
     >                    D1*(0.5D0*COEFF(3)*1.D-2 +
     >                        D1*(COEFF(4)*1.D-3/6.D0 +
     >                            D1*(COEFF(5)*1.D-3/12.D0))))
        pb1_1=COEFF(6) + D1*(COEFF(7)*1.D-1 +
     >                    D1*(0.5D0*COEFF(8)*1.D-2 +
     >                        D1*(COEFF(9)*1.D-3/6.D0 +
     >                            D1*(COEFF(10)*1.D-3/12.D0))))
        pC0_1=COEFF(11) + D1*(COEFF(12)*1.D-1 +
     >                    D1*(0.5D0*COEFF(13)*1.D-2 +
     >                        D1*(COEFF(14)*1.D-3/6.D0 +
     >                            D1*(COEFF(15)*1.D-3/12.D0))))
        pC1_1=COEFF(16) + D1*(COEFF(17)*1.D-1 +
     >                    D1*(0.5D0*COEFF(18)*1.D-2 +
     >                        D1*(COEFF(19)*1.D-3/6.D0 +
     >                            D1*(COEFF(20)*1.D-3/12.D0))))
        pD_MX3=COEFF(21) + D1*(COEFF(22)*1.D-1 +
     >                    D1*(0.5D0*COEFF(23)*1.D-2 +
     >                        D1*(COEFF(24)*1.D-3/6.D0 +
     >                            D1*(COEFF(25)*1.D-3/12.D0))))
        pD_MX4=COEFF(26) + D1*(COEFF(27)*1.D-1 +
     >                    D1*(0.5D0*COEFF(28)*1.D-2 +
     >                        D1*(COEFF(29)*1.D-3/6.D0 +
     >                            D1*(COEFF(30)*1.D-3/12.D0))))
        pD_MX5=COEFF(31) + D1*(COEFF(32)*1.D-1 +
     >                    D1*(0.5D0*COEFF(33)*1.D-2 +
     >                        D1*(COEFF(34)*1.D-3/6.D0 +
     >                            D1*(COEFF(35)*1.D-3/12.D0))))
C
C
        pb0_2=COEFF(51) + D1*(COEFF(52)*1.D-1 +
     >                    D1*(0.5D0*COEFF(53)*1.D-2 +
     >                        D1*(COEFF(54)*1.D-3/6.D0 +
     >                            D1*(COEFF(55)*1.D-3/12.D0))))
        pb1_2=COEFF(56) + D1*(COEFF(57)*1.D-1 +
     >                    D1*(0.5D0*COEFF(58)*1.D-2 +
     >                        D1*(COEFF(59)*1.D-3/6.D0 +
     >                            D1*(COEFF(60)*1.D-3/12.D0))))
        pC0_2=COEFF(61) + D1*(COEFF(62)*1.D-1 +
     >                    D1*(0.5D0*COEFF(63)*1.D-2 +
     >                        D1*(COEFF(64)*1.D-3/6.D0 +
     >                            D1*(COEFF(65)*1.D-3/12.D0))))
        pC1_2=COEFF(66) + D1*(COEFF(67)*1.D-1 +
     >                    D1*(0.5D0*COEFF(68)*1.D-2 +
     >                        D1*(COEFF(69)*1.D-3/6.D0 +
     >                            D1*(COEFF(70)*1.D-3/12.D0))))
        pD_NX3=COEFF(71) + D1*(COEFF(72)*1.D-1 +
     >                    D1*(0.5D0*COEFF(73)*1.D-2 +
     >                        D1*(COEFF(74)*1.D-3/6.D0 +
     >                            D1*(COEFF(75)*1.D-3/12.D0))))
        pD_NX4=COEFF(76) + D1*(COEFF(77)*1.D-1 +
     >                    D1*(0.5D0*COEFF(78)*1.D-2 +
     >                        D1*(COEFF(79)*1.D-3/6.D0 +
     >                            D1*(COEFF(80)*1.D-3/12.D0))))
        pD_NX5=COEFF(36) + D1*(COEFF(37)*1.D-1 +
     >                    D1*(0.5D0*COEFF(38)*1.D-2 +
     >                        D1*(COEFF(39)*1.D-3/6.D0 +
     >                            D1*(COEFF(40)*1.D-3/12.D0))))
C
        pD_MNX2=COEFF(81) + D1*(COEFF(82)*1.D-1 +
     >                    D1*(0.5D0*COEFF(83)*1.D-2 +
     >                        D1*(COEFF(84)*1.D-3/6.D0 +
     >                            D1*(COEFF(85)*1.D-3/12.D0))))
        pD_MNX3=COEFF(86) + D1*(COEFF(87)*1.D-1 +
     >                    D1*(0.5D0*COEFF(88)*1.D-2 +
     >                        D1*(COEFF(89)*1.D-3/6.D0 +
     >                            D1*(COEFF(90)*1.D-3/12.D0))))
C
C
      RETURN
      END Subroutine
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE OSMCALC(mCaCl2,mCa,mCaCl,mCl,DHpara,
     >                   b0_1,b1_1,C0_1,C1_1,
     >                   b0_2,b1_2,C0_2,C1_2,
     >                   D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                   D_MNX2,D_MNX3,
     >                   alpha1,omega1,alpha2,omega2,
     >                   theta,psi,osmstoic)
      IMPLICIT REAL*8 (A-H,O-Z)
C
      REAL*8 mCa,mCaCl,mCl,mCaCl2,Ix
C
      Ix=0.5D0*(4*mCa + mCaCl + mCl)
      SQRTIX=SQRT(Ix)
      ZFUNC=2*mCa + mCaCl + mCl
      etheta=EFUNC(1,2,DHpara,Ix)
      edtheta=EDFUNC(1,2,DHpara,Ix,etheta)
C
      Bphi1=b0_1 + b1_1*EXP(-alpha1*SQRTIX)
      Bphi2=b0_2 + b1_2*EXP(-alpha2*SQRTIX)
      CTphi1=C0_1+C1_1*EXP(-omega1*SQRTIX)
      CTphi2=C0_2+C1_2*EXP(-omega2*SQRTIX)
      DH=-DHpara*Ix**1.5D0/(1.D0 + 1.2D0*SQRTIX)
      DUM1=mCa*mCl*(Bphi1 + ZFUNC*CTphi1)
     >    +mCaCl*mCl*(Bphi2 + ZFUNC*CTphi2)
     >    +mCa*mCaCl*(theta + etheta + edtheta*Ix + mCl*psi)
      SUMMI=mCa+mCl+mCaCl
C
C   ..the extended osmotic coefficient equation:
      osmcoef=1.D0+(2.D0/SUMMI)*(DH + DUM1) + 1.D0/SUMMI *
     >    (3*mCa*mCl**3*D_MX3 + 4*mCa*mCl**4*D_MX4
     >    +  5*mCa*mCl**5*D_MX5
     >    +  3*mCaCl*mCl**3*D_NX3 + 4*mCaCl*mCl**4*D_NX4
     >    +  5*mCaCl*mCl**5*D_NX5
     >    +  3*mCa*mCaCl*mCl**2*D_MNX2 + 4*mCa*mCaCl*mCl**3*D_MNX3)
C
C   ..convert to stoichiometric value:
      osmstoic=osmcoef*(mCa+mCaCl+mCl)/(3*mCaCl2)
C
      RETURN
      END Subroutine
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE ACTCOEF(DHpara,mCa,mCaCl,mCl,
     >                   b0_1,b1_1,C0_1,C1_1,
     >                   b0_2,b1_2,C0_2,C1_2,
     >                   D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                   D_MNX2,D_MNX3,
     >                   alpha1,omega1,alpha2,omega2,
     >                   theta,psi,
     >                   lnactCa,lnactCaCl,lnactCl)
      IMPLICIT REAL*8 (A-H,O-Z)
C
      REAL*8 DHpara,Ix,mCa,mCl,mCaCl,
     >       lnactCa,lnactCaCl,lnactCl
C
C -------------------------------------------------------------------
C
C  b0_1, b1_1, C0_1, C1_1, D_MX3, D_MX4, D_MX5 = Ca - Cl interactions
C
C  b0_2, b1_2, C0_2, C1_2, D_NX3, D_NX4, D_NX5 = CaCl - Cl interactions
C
C  theta, psi = Ca - CaCl, and Ca-CaCl-Cl interactions
C
C  D_MNX2, D_MNX3 = Ca-CaCl-Cl interactions
C
C  alpha1, omega1 = are for Ca - Cl interations
C  alpha2, omega2 = are for CaCl - Cl interations
C
C -------------------------------------------------------------------
C
C
      zCa=2.D0
      zCaCl=1.D0
      zCl=1.D0
C
      Ix=0.5D0*(4*mCa + mCl + mCaCl)
      ZFUNC=mCa*2 + mCaCl + mCl
      SQRTIX=SQRT(Ix)
      dum1=1.D0+1.2D0*SQRTIX
C
c..      write(2,'(1x,''ACTCOEF 1 '',10(E11.4,1X))') mCa,mCaCl,mCl
C
      dumh1=HFUNC(omega1,SQRTIX)
      dumh2=HFUNC(omega2,SQRTIX)
      dumhd1=EXP(-omega1*SQRTIX)/2.D0 - 2*dumh1
      dumhd2=EXP(-omega2*SQRTIX)/2.D0 - 2*dumh2
C
c..      write(2,'(1x,''ACTCOEF 2 '',10(E11.4,1X))') dumh1,dumh2,dumhd1,
c..     >                                            dumhd2
C
      etheta=EFUNC(1,2,DHpara,Ix)
      edtheta=EDFUNC(1,2,DHpara,Ix,etheta)
C
      B1=b0_1 + b1_1*GFUNC(alpha1,SQRTIX)
      B2=b0_2 + b1_2*GFUNC(alpha2,SQRTIX)
      CT1=C0_1 + 4*C1_1*dumh1
      CT2=C0_2 + 4*C1_2*dumh2
C
      BD1=b1_1*GDFUNC(alpha1,SQRTIX)/Ix
      BD2=b1_2*GDFUNC(alpha2,SQRTIX)/Ix
      CTD1=4*C1_1*dumhd1/Ix
      CTD2=4*C1_2*dumhd2/Ix
C
      F=-DHpara*(SQRTIX/dum1 + 2.D0/1.2D0*LOG(dum1))
     >  +mCa*mCl*(BD1 + ZFUNC*CTD1/2)
     >  +mCaCl*mCl*(BD2 + ZFUNC*CTD2/2)
     >  +mCa*mCaCl*edtheta
C
C
C
      lnactCa= 4*F + mCl*(2*B1 + ZFUNC*CT1)
     >        +mCaCl*(2*(theta+etheta) + mCl*psi)
     >        +zCa*(mCa*mCl*CT1 + mCaCl*mCl*CT2)
C
     >        + mCl**3*D_MX3 + mCl**4*D_MX4 + mCl**5*D_MX5
     >        + mCaCl*mCl**2*D_MNX2 + mCaCl*mCl**3*D_MNX3
C.....actCa=EXP(lnactCa)
C     ------------------
C
      lnactCaCl= F + mCl*(2*B2 + ZFUNC*CT2)
     >         +mCa*(2*(theta+etheta) + mCl*psi)
     >         +zCaCl*(mCa*mCl*CT1 + mCaCl*mCl*CT2)
C
     >        + mCl**3*D_NX3 + mCl**4*D_NX4 + mCl**5*D_NX5
     >        + mCa*mCl**2*D_MNX2 + mCa*mCl**3*D_MNX3
C......actCaCl=EXP(lnactCaCl)
C      ----------------------
C
      lnactCl= F + mCa*(2*B1 + ZFUNC*CT1)
     >        + mCaCl*(2*B2 + ZFUNC*CT2)
     >        + mCa*mCaCl*psi
     >        + zCl*(mCa*mCl*CT1 + mCaCl*mCl*CT2)
C
     > + 3*mCa*mCl**2*D_MX3 + 4*mCa*mCl**3*D_MX4 + 5*mCa*mCl**4*D_MX5
     > + 3*mCaCl*mCl**2*D_NX3 + 4*mCaCl*mCl**3*D_NX4
     > + 5*mCaCl*mCl**4*D_NX5
     > + 2*mCa*mCaCl*mCl*D_MNX2 + 3*mCa*mCaCl*mCl**2*D_MNX3
C.....actCl=EXP(lnactCl)
C     ------------------
C
C
      RETURN
      END Subroutine
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION FC05(DHpara,mCaCl2,mCafree,const,b0_1,b1_1,C0_1,C1_1,
     >                                          b0_2,b1_2,C0_2,C1_2,
     >                          D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                                                D_MNX2,D_MNX3,
     >                                  alpha1,omega1,alpha2,omega2,
     >                                                    theta,psi,
     >                                    lnactCa,lnactCaCl,lnactCl,
     >                                                mCa,mCaCl,mCl)
      IMPLICIT REAL*8(A-H,O-Z)
C
      REAL*8 Kstar,mCaCl2,mCafree,mCa,mCaCl,mCl,lnactCa,lnactCaCl,
     >       lnactCl
C
      IF(mCafree .EQ. 0.D0) THEN
        FC05=mCaCl2
        RETURN
      ENDIF
C
      mCa=mCafree
      mCaCl=mCaCl2-mCafree
      mCl=2*mCaCl2-mCaCl
C
C....................................................................
C
C     equilibrium constant is calculated in ZBRENT
C
C....................................................................
C
      CALL ACTCOEF(DHpara,mCa,mCaCl,mCl,b0_1,b1_1,C0_1,C1_1,
     >                                  b0_2,b1_2,C0_2,C1_2,
     >                          D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                                                D_MNX2,D_MNX3,
     >                                  alpha1,omega1,alpha2,omega2,
     >                                                    theta,psi,
     >                                    lnactCa,lnactCaCl,lnactCl)
C
C
C     association constant: Ca + Cl = CaCl
C
      aprodln=lnactCa-lnactCaCl+lnactCl
      IF(aprodln.GT.700.D0) THEN
C     ..this is to prevent the activity product going above numerical
C       limits.
        Kstar= CONST * EXP(700.D0)
      ELSE
        Kstar= CONST * EXP(aprodln)
      ENDIF
C
C     now function F
C
C   ..total Ca - free Ca - assoc. Ca:
      FC05=mCaCl2 - mCa*(1.D0 + mCl*Kstar)
C
c.....WRITE(*,'(1X,''F2 = '',5(E11.4,1X))') FC05
C
      END Function
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION ZBRENT(FC05,LOW,HIGH,CONST,DHpara,mCaCl2,
     >                b0_1,b1_1,C0_1,C1_1,
     >                b0_2,b1_2,C0_2,C1_2,
     >                D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                D_MNX2,D_MNX3,
     >                alpha1,omega1,alpha2,omega2,
     >                theta,psi,
     >                lnactCa,lnactCaCl,lnactCl)
C
      IMPLICIT REAL*8(A-H,O-Z)
C
      PARAMETER(ITMAX=50, TOL=1.D-15, EPS=1.5D-16)
C
      REAL*8 mCaCl2,LOW,mCa,mCaCl,mCl,lnactCa,lnactCaCl,lnactCl
C
      EXTERNAL FC05
C
C------------------------------------------------------------
C     !subroutine contains several 'IF's where real*8 variable
C     is tested for being = 0.D0
C
C     answer = ZBRENT = free Ca++
C
C------------------------------------------------------------
C
      A=LOW
      B=HIGH
C
C
      FB=FC05(DHpara,mCaCl2,B,const,b0_1,b1_1,C0_1,C1_1,
     >                              b0_2,b1_2,C0_2,C1_2,
     >              D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                                    D_MNX2,D_MNX3,
     >                      alpha1,omega1,alpha2,omega2,
     >                                        theta,psi,
     >                        lnactCa,lnactCaCl,lnactCl,
     >                                    mCa,mCaCl,mCl)
C
C   ..if the function value is within machine accuracy of zero then return:
      IF(ABS(FB).LE.EPS) THEN
        ZBRENT=B
        RETURN
      ENDIF
C
      FA=FC05(DHpara,mCaCl2,A,const,b0_1,b1_1,C0_1,C1_1,
     >                              b0_2,b1_2,C0_2,C1_2,
     >              D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                                    D_MNX2,D_MNX3,
     >                      alpha1,omega1,alpha2,omega2,
     >                                        theta,psi,
     >                        lnactCa,lnactCaCl,lnactCl,
     >                                    mCa,mCaCl,mCl)
C
      IF(ABS(FA).LE.EPS) THEN
        ZBRENT=A
        RETURN
      ENDIF

C
C
C
C   ..check bracketting:
      IF(FA*FB.GT.0.D0) THEN
        WRITE(2,'(1X,''MUST BRACKET ROOT! STOP IN ZBRENT.'')')
        WRITE(2,'(1X,''VALUES: '',2(E11.4,1X,E11.4))') A,B,FA,FB
        STOP
      ENDIF
C
C
C
      C=B
      FC=FB
      DO 2 ITER=1,ITMAX
C.......WRITE(2,'(1X,''ITER'',I2)') ITER
        IF((FB.GT.0.D0 .AND. FC.GT.0.D0) .OR.
     >     (FB.LT.0.D0 .AND. FC.LT.0.D0)) THEN
          C=A
          FC=FA
          D=B-A
          E=D
        ENDIF
        IF(ABS(FC) .LT. ABS(FB)) THEN
          A=B
          B=C
          C=A
          FA=FB
          FB=FC
          FC=FA
        ENDIF
C
        TOL1=2.D0*EPS*ABS(B)+5.0D-1*TOL
        XM=5.0D-1*(C-B)
        IF((ABS(XM).LE.TOL1) .OR. (FB.EQ.0.D0)) THEN
          ZBRENT=B
          RETURN
        ENDIF
C
        IF((ABS(E).GE.TOL1) .AND. (ABS(FA).GT.ABS(FB))) THEN
          S=FB/FA
          IF(A.EQ.C) THEN
            P=2.D0*XM*S
            Q=1.D0-S
          ELSE
            Q=FA/FC
            R=FB/FC
            P=S*(2.D0*XM*Q*(Q-R)-(B-A)*(R-1.D0))
            Q=(Q-1.D0)*(R-1.D0)*(S-1.D0)
          ENDIF
          IF(P .GT. 0.D0) Q=-Q
          P=ABS(P)
          IF(2.D0*P .LT. MIN(3.D0*XM*Q-ABS(TOL1*Q),ABS(E*Q))) THEN
            E=D
            D=P/Q
          ELSE
            D=XM
            E=D
          ENDIF
        ELSE
          D=XM
          E=D
        ENDIF
        A=B
        FA=FB
        IF(ABS(D).GT.TOL1) THEN
          B=B+D
        ELSE
          B=B+SIGN(TOL1,XM)
        ENDIF
C
      FB=FC05(DHpara,mCaCl2,B,const,b0_1,b1_1,C0_1,C1_1,
     >                              b0_2,b1_2,C0_2,C1_2,
     >              D_MX3,D_MX4,D_MX5,D_NX3,D_NX4,D_NX5,
     >                                    D_MNX2,D_MNX3,
     >                      alpha1,omega1,alpha2,omega2,
     >                                        theta,psi,
     >                        lnactCa,lnactCaCl,lnactCl,
     >                                    mCa,mCaCl,mCl)
C
C
c..   WRITE(*,'(1X,''FMAIN'',E11.4,1X,E11.4)') B,FB
C
2     CONTINUE
C
      WRITE(2,200) ITMAX
      STOP
C
C
200   FORMAT(1X/' !ERROR: ITERATION LIMIT REACHED DURING FULL CALCS.'/
     >          ' ITMAX2= ',I2/
     >          ' *** TERMINATED IN ZBRENT ***')
C
      END Function
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      END MODULE
