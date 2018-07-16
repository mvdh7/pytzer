      MODULE PitzH2SO4
C
C     +++++++
!      PRIVATE
C     +++++++
C     ++++++
      PUBLIC :: H2SO4
C     ++++++
C
C
        CONTAINS
C
C
      Subroutine H2SO4(T, mH2SO4, b0HHSO4, b1HHSO4, C0HHSO4, C1HHSO4,
     >                 b0HSO4, b1HSO4, C0HSO4, C1HSO4, alphaHSO4, KHSO4,
     >                 mH, mHSO4, mSO4, OSM, lnGamma)
C
      IMPLICIT REAL*8 (A-H,O-Z)
C
C     ----------------------------------------------------------
C     KHSO4 is the thermodynamic dissociation constant of HSO4-.
C
C     The C parameters that are used internally are Cphi, and are
C     related to the arguments by:
C
C     H-HSO4: Cphi = C * 2
C     H-SO4 : Cphi = C * 2*SQRT(2)
C     -----------------------------------------------------------
C
C     Declare f2py intentions
!f2py intent(in) T, mH2SO4, b0HHSO4, b1HHSO4, C0HHSO4, C1HHSO4
!f2py intent(in) b0HSO4, b1HSO4, C0HSO4, C1HSO4, alphaHSO4, KHSO4
!f2py intent(out) mH, mHSO4, mSO4, OSM, lnGamma
C
C
      REAL*8, PARAMETER :: EPS = EPSILON(1.D0)
C
C
!  -- arguments --
      REAL*8, INTENT(IN) :: T, mH2SO4, b0HHSO4, b1HHSO4, C0HHSO4,
     >                      C1HHSO4, b0HSO4, b1HSO4, C0HSO4, C1HSO4,
     >                      KHSO4
      REAL*8, INTENT(OUT) ::mH, mHSO4, mSO4, OSM, lnGamma
C
!  -- local --
      REAL*8 :: HIGH, LOW, B(5,5,3), C(5,5)
      REAL*8 :: THETAC(5,5) = 0.D0, THETAA(5,5) = 0.D0,
     >          PSIC(5,5,5) = 0.D0, PSIA(5,5,5) = 0.D0
C
      SAVE THETAC, THETAA, PSIC, PSIA
C
C
C
C
!   ..assign parameters
!     -----------------
      B(:,:,:) = 0.D0
      C(:,:)   = 0.D0
C
!   ..H-HSO4
!     ------
      B(1,1,1) = b0HHSO4
      B(1,1,2) = b1HHSO4
      C(1,1)   = C0HHSO4 * 2
      C(2,1)   = C1HHSO4 * 2
C
!   ..H-SO4
!     -----
      Factor = 2.D0*SQRT(2.D0)
      B(1,2,1) = b0HSO4
      B(1,2,2) = b1HSO4
      C(1,2)   = C0HSO4 * Factor
      C(2,2)   = C1HSO4 * Factor
C
      LOW  = mH2SO4
      HIGH = 2*mH2SO4
C
      mH = ZBRENT(FC05, LOW, HIGH, T, mH2SO4,
     >            0.D0, 0.D0, EPS, KHSO4,
     >            B, C, alphaHSO4, THETAC, THETAA, PSIC, PSIA,
     >            actH, actHSO4, actSO4, OSM, AWLG)

!   ..results
!     -------
      mHSO4 = 2*mH2SO4 - mH
      mSO4  = mH2SO4 - mHSO4
C
      OSM = -AWLG/(0.0180152D0*3.D0*mH2SO4)
      dum = actH**2 * actSO4 * mH**2 * mSO4 / (4*mH2SO4**3)
      lnGamma = LOG(dum)/3.D0  ! mean act. coef.
C
C     ******
      RETURN
C     ******
C
      END Subroutine
C
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION ZBRENT(FC05, LOW, HIGH, TEMP, MTOT,
     >                MHG1, MPB1, EPS, CONST,
     >                BCOEF, CCOEF, alphaHSO4,
     >                THETAC, THETAA, PSIC, PSIA,
     >                ACTH, ACTHSO4, ACTSO4, OSM, AWLG)
      IMPLICIT REAL*8(A-H,O-Z)
C
      REAL*8 :: T_old = -999.D0
C
      PARAMETER(ITMAX=50, TOL=1.D-16, EPS_Temperature=1.D-12,
     >          NCOEF=80,TR=298.15D0,RG=8.3144D0,P1=0.0128310991D0,
     >          P2=-9.486301516D0,P3=1962.6177D0)
C
      EXTERNAL FC05

      LOGICAL :: TDifferent
C
      REAL*8 MTOT,MHG1,MPB1,BCOEF(5,5,3),CCOEF(5,5),
     >       THETAC(5,5),THETAA(5,5),PSIC(5,5,5),PSIA(5,5,5),LOW
C
      SAVE Told, APHI
C
C------------------------------------------------------------
C     !subroutine contains several 'IF's where real*8 variable
C     is tested for being = 0.D0
C
C------------------------------------------------------------
C
C
!   ..only calculate Aphi when the temperature differs from
!     the last call (otherwise uses the saved value)
!     -----------------------------------------------------
      TDifferent = ABS(TEMP - T_old) .GT. EPS_Temperature
      T_Old = TEMP
      IF(TDifferent) APHI = AFT(TEMP)
C
C...................................................................
C
      A = LOW
      B = HIGH
C
      FA=FC05(A,TEMP,MTOT,MHG1,MPB1,CONST,APHI,
     >              BCOEF,CCOEF,alphaHSO4,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      FB=FC05(B,TEMP,MTOT,MHG1,MPB1,CONST,APHI,
     >              BCOEF,CCOEF,alphaHSO4,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
C
      IF(FA*FB.GT.0.D0) THEN
        WRITE(2,'(1X,''MUST BRACKET ROOT! STOP IN ZBRENT.'')')
        WRITE(2,'(1X,''VALUES: '',2(E11.4,1X,E11.4))') A,B,FA,FB
        STOP
      ENDIF
C
      C=B
      FC=FB
      DO 2 ITER=1,ITMAX
C.......WRITE(2,'(1X,''ITER'',I2)') ITER
        ITERC2=ITER
        IF(FB*FC .GT. 0.D0) THEN
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
        FB=FC05(B,TEMP,MTOT,MHG1,MPB1,CONST,APHI,
     >              BCOEF,CCOEF,alphaHSO4,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
C
C.......WRITE(2,'(1X,''FMAIN'',E11.4,1X,E11.4)') B,FB
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
      END FUNCTION
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION AFT(T)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 AA(0:18), TX(0:18)
C
      DATA AA/0.797256081240D+00,0.573389669896D-01,
     > 0.977632177788D-03, 0.489973732417D-02,-0.313151784342D-02,
     > 0.179145971002D-02,-0.920584241844D-03, 0.443862726879D-03,
     >-0.203661129991D-03, 0.900924147948D-04,-0.388189392385D-04,
     > 0.164245088592D-04,-0.686031972567D-05, 0.283455806377D-05,
     >-0.115641433004D-05, 0.461489672579D-06,-0.177069754948D-06,
     > 0.612464488231D-07,-0.175689013085D-07/
     > XMIN,XMAX/0.234150D+03,0.373150D+03/
C
C     ------------------------------------------------
C   ..for 234.15 < T < 373.15 K, polynomial reproduces
C     Archer's values directly:
C     ------------------------------------------------
      X=(2.D0*T-XMAX-XMIN)/(XMAX-XMIN)
      TX(0)=1.D0
      TX(1)=X
C
      N = 17
C
      AFT=0.5*AA(0)*TX(0) + AA(1)*TX(1)
      DO 1 I=1,N
        TX(I+1)=2*X*TX(I) - TX(I-1)
        AFT=AFT + AA(I+1)*TX(I+1)
1     CONTINUE
C
      END Function
C
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION FC05(MH,TEMP,MTOT,MHG1,MPB1,CONST,APHI,
     >              B,C,alphaHSO4,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      IMPLICIT REAL*8 (A-H,O-Z)
C
      PARAMETER(NV=1000,
     >          NCOEF=80,
     >          R=8.3144D0)
C
      REAL*8 MTOT,MH,MHSO4,MSO4,KSTAR,MHG1,MPB1
      REAL*8 CATM(5),ANM(5),NEUTM(5),
     >       B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),PSIC(5,5,5),
     >       PSIA(5,5,5)
C
C....................................................................
C
C     equilibrium constant and APHI are calculated in ZBRENT
C
C....................................................................
C     nb: mtot = total H2SO4 concentration
C
      SO4T=MTOT+MPB1+MHG1
C
      MHSO4=MTOT*2.D0 - MH
      MSO4=MTOT-MHSO4+MPB1+MHG1
C
      CATM(1)=MH
      CATM(2)=2.D0*MHG1
      CATM(3)=MPB1
      ANM(1)=MHSO4
      ANM(2)=MSO4
C
      XION=0.5D0*(CATM(1)+CATM(2)+4.D0*CATM(3)+ANM(1)+4.D0*ANM(2))
C
C.....CALL FOR SIMPLIFIED EQUATIONS .............................
C
      CALL ACT(APHI,B,C,THETAC,THETAA,PSIC,PSIA,CATM,ANM,NEUTM,XION,
     >         ACTH,ACTHSO4,ACTSO4,OSM,AWLG,TEMP,alphaHSO4)
C
C.....WRITE(2,'(1X,''MH,MHSO4,MSO4 '',6(E11.4,2X))') MH,MHSO4,MSO4
C
      KSTAR=CONST*ACTH*ACTSO4/ACTHSO4
C
C     now function F (revised)
C
      FC05=MH*SO4T/(1.D0/KSTAR + MH) - MHSO4
C
C.....old function below..
C.....FC05=MH*(1.D0+(MSO4+MHSO4)*KSTAR/(1.D0+MH*KSTAR)) - (MH+MHSO4)
C
C      WRITE(*,'(1X,''F = '',4(E11.4,2X))') FC05,MTOT,MH,T
C      WRITE(2,'(1X,15(E10.3,1X)') MH,MTOT,TEMP,ACTH,ACTHSO4,
C     > ACTSO4,OSM,AWLG
C
      END FUNCTION
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE ACT(A,B,C,THETAC,THETAA,PSIC,PSIA,CATM,ANM,NEUTM,
     >               XION,ACTH,ACTHSO4,ACTSO4,OSM,AWLG,TEMP,alphaHSO4)
C
      IMPLICIT REAL*8(A-H,O-Z)
C
      REAL*8 CATM(5),ANM(5),NEUTM(5),MH,MHG,MPB,MHSO4,MSO4,
     >       B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),PSIC(5,5,5),
     >       PSIA(5,5,5)
C
      ALPHA1=2.D0
      ALPHAX=alphaHSO4
      ALPHA2=2.5D0
      Z=2.D0*(ANM(1)+2.D0*ANM(2))
      CDUM=(2.D0*SQRT(2.D0))
      SQXION=SQRT(XION)
      EXPASQ=EXP(-ALPHA2*SQXION)
C
      BF11=B(1,1,1)+B(1,1,2)*GFUNC(ALPHA1,XION)
      BPF11=B(1,1,1)+B(1,1,2)*EXP(-ALPHA1*SQXION)
      BF12=B(1,2,1)+B(1,2,2)*GFUNC(ALPHAX,XION)
      BPF12=B(1,2,1)+B(1,2,2)*EXP(-ALPHAX*SQXION)
      ETHA=EFUNC(1,2,A,XION)
      EDTHA=EDFNCS(1,2,A,XION,ETHA)
      FF=FFUNCS(CATM,ANM,A,B,C,THETAC,THETAA,PSIC,PSIA,XION,EDTHA,
     >          TEMP,alphaHSO4)
C
      MH=CATM(1)
      MHG=CATM(2)
      MPB=CATM(3)
      MHSO4=ANM(1)
      MSO4=ANM(2)
C
C//////////////////////////////////////////////////////////////////////
      EXTRA=(-12.D0+(12.D0+ALPHA2**4*XION**2/2.D0+2.D0*ALPHA2**3*
     >       XION**(1.5D0)+6.D0*ALPHA2**2*XION+12.D0*ALPHA2*SQXION)
     >       *EXPASQ )/(XION**3*ALPHA2**4)
      EXTRA2=(6.D0-(6.D0+6.D0*ALPHA2*SQXION+
     >       3.D0*ALPHA2**2*XION+ALPHA2**3*XION**1.5D0)
     >       *EXPASQ)/(ALPHA2**4*XION**2)
      CTOT11=C(1,1)+4.D0*C(2,1)*EXTRA2
      CTOT12=C(1,2)+4.D0*C(2,2)*EXTRA2
C//////////////////////////////////////////////////////////////////////
C
C===== activity coefficient of H+ ===================================
C
C//////////////////////////////////////////////////////////////////////
      CSUMCA=MH*(MHSO4*CTOT11/2.D0 + MSO4*CTOT12/CDUM)
     >      +MH*(MHSO4*Z*2.D0*C(2,1)*EXTRA/2.D0)
     >      +MH*(MSO4*Z*2.D0*C(2,2)*EXTRA/CDUM)
      CSUMA=MHSO4*(2.D0*BF11 + Z*CTOT11/2.D0)
     >     +MSO4*(2.D0*BF12 + Z*CTOT12/CDUM)
C//////////////////////////////////////////////////////////////////////
C
c.....CSUMCA=MH*(MHSO4*C(1,1)/2.D0 + MSO4*C(1,2)/CDUM)
C
c.... CSUMA=MHSO4*(2.D0*BF11 + Z*C(1,1)/2.D0)
c....>     +MSO4*(2.D0*BF12 + Z*C(1,2)/CDUM)
C
      CSUMAA=MHSO4*MSO4*PSIA(1,2,1)
C
      CSUMC=MPB*2.D0*ETHA
C
      ACTH=EXP(FF+CSUMA+CSUMC+CSUMAA+CSUMCA)
C
C===== activity coefficient of HSO4 =================================
C
C////////////////////////////////////////////////////////////////////
      ASUMCA=MH*(MHSO4*CTOT11/2.D0 + MSO4*CTOT12/CDUM)
     >      +MH*(MHSO4*Z*2.D0*C(2,1)*EXTRA/2.D0)
     >      +MH*(MSO4*Z*2.D0*C(2,2)*EXTRA/CDUM)
      ASUMC=MH*(2.D0*BF11 + Z*CTOT11/2.D0)
C////////////////////////////////////////////////////////////////////
C
c.....ASUMCA=MH*(MHSO4*C(1,1)/2.D0 + MSO4*C(1,2)/CDUM)
C
c.....ASUMC=MH*(2.D0*BF11 + Z*C(1,1)/2.D0)
C
      ASUMA=MSO4*(2.D0*(THETAA(1,2)+ETHA) + MH*PSIA(1,2,1))
C
      ACTHSO4=EXP(FF+ASUMC+ASUMA+ASUMCA)
C
C===== activity coefficient of SO4 =================================
C////////////////////////////////////////////////////////////////////
      ASUMCA=2.D0*MH*(MHSO4*CTOT11/2.D0 + MSO4*CTOT12/CDUM)
     >      +2.D0*MH*(MHSO4*Z*4.D0*C(2,1)*EXTRA/2.D0)
     >      +2.D0*MH*(MSO4*Z*4.D0*C(2,2)*EXTRA/CDUM)
      ASUMC=MH*(2.D0*BF12 + Z*CTOT12/CDUM)
C////////////////////////////////////////////////////////////////////
C
c.....ASUMCA=2.D0*MH*(MHSO4*C(1,1)/2.D0 + MSO4*C(1,2)/CDUM)
C
c.....ASUMC=MH*(2.D0*BF12 + Z*C(1,2)/CDUM)
C
      ASUMA=MHSO4*(2.D0*(THETAA(1,2)+ETHA) + MH*PSIA(1,2,1))
C
      ACTSO4=EXP(4.D0*FF+ASUMC+ASUMA+ASUMCA)
C
C==== osmotic coefficient ==========================================
C
C////////////////////////////////////////////////////////////////////
       SUMCA=MH*MHSO4*(BPF11 + Z*C(1,1)/2.D0)
     >       +MH*MSO4*(BPF12 + Z*C(1,2)/CDUM)
     >      +MH*MHSO4*(Z*C(2,1)*EXPASQ/2.D0)
     >       +MH*MSO4*(Z*C(2,2)*EXPASQ/CDUM)
C///////////////////////////////////////////////////////////////////
C
c....  SUMCA=MH*MHSO4*(BPF11 + Z*C(1,1)/2.D0)
c....>       +MH*MSO4*(BPF12 + Z*C(1,2)/CDUM)
C
       SUMCC=(ETHA+XION*EDTHA)*(MH*MPB + MHG*MPB)
C
       SUMAA=MHSO4*MSO4*(THETAA(1,2)+ETHA+XION*EDTHA + MH*PSIA(1,2,1))
C
       SUMMOL=MH+MPB+MHG+MSO4+MHSO4
       OSM=1.D0+2.D0/SUMMOL*(-A*XION**1.5D0/(1.D0+1.2D0*SQXION)
     >                       +SUMCA+SUMCC+SUMAA)
       AWLG=-1.80152D-2*OSM*SUMMOL
C
       RETURN
C
       END SUBROUTINE
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION FFUNCS(CATM,ANM,A,B,C,THETAC,THETAA,PSIC,PSIA,XION,EDTHA,
     >                TEMP,alphaHSO4)
       IMPLICIT REAL*8(A-H,O-Z)
       REAL*8 CATM(5),ANM(5)
       REAL*8 B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),PSIC(5,5,5),
     >        PSIA(5,5,5)
C
       DSQ=SQRT(XION)
       DUM=-A*(DSQ/(1.D0+1.2D0*DSQ)+(2.D0/1.2D0)*LOG(1.D0+1.2D0*DSQ))
       ALPHA1=2.D0
       ALPHAX=alphaHSO4
C
       SUMCA=CATM(1)*ANM(1)*B(1,1,2)*GDFUNC(ALPHA1,XION)/XION
     >      +CATM(1)*ANM(2)*B(1,2,2)*GDFUNC(ALPHAX,XION)/XION
C
       SUMCCAA=EDTHA*(CATM(1)*CATM(3)+CATM(2)*CATM(3)+ANM(1)*ANM(2))
C
       FFUNCS=DUM+SUMCA+SUMCCAA
C
      END Function
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION GFUNC(ALPHA,XION)
       IMPLICIT REAL*8(A-H,O-Z)
       DUM=ALPHA*SQRT(XION)
       GFUNC=2.D0*(1.D0-(1.D0+DUM)*EXP(-DUM))/DUM**2
      END FUNCTION
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION GDFUNC(ALPHA,XION)
       IMPLICIT REAL*8(A-H,O-Z)
       DUM=ALPHA*SQRT(XION)
       GDFUNC=-2.D0*(1.D0-(1.D0+DUM+5.D-1*DUM**2)*EXP(-DUM))/DUM**2
      END FUNCTION
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION EFUNC(ICHARG,JCHARG,A,XION)
       IMPLICIT REAL*8(A-H,O-Z)
       REAL*8 XIJ(3),J0(3)
C
       IF((ICHARG.EQ.JCHARG) .OR. (XION.LE.1.D-30)) THEN
         EFUNC=0.D0
       ELSE
         DUM=6.D0*A*SQRT(XION)
         XIJ(1)=ICHARG*JCHARG*DUM
         XIJ(2)=ICHARG**2*DUM
         XIJ(3)=JCHARG**2*DUM
         DO 1 I=1,3
           J0(I)=XIJ(I)/(4.D0+4.581D0*XIJ(I)**(-7.238D-1)*EXP(-1.2D-2*
     >           XIJ(I)**5.28D-1))
1        CONTINUE
         EFUNC=ICHARG*JCHARG/(4.D0*XION)*(J0(1)-5.D-1*J0(2)-5.D-1*J0(3))
       ENDIF
      END FUNCTION
C
C+++++ function below is for 'simplified' equations ++++++++++
C
      FUNCTION EDFNCS(ICHARG,JCHARG,A,XION,ETHA)
       IMPLICIT REAL*8(A-H,O-Z)
       REAL*8 XIJ(3),J1(3)
C
       IF((ICHARG.EQ.JCHARG) .OR. (XION.LE.1.D-30)) THEN
          EDFUNC=0.D0
       ELSE
          DUM1=6.D0*A*SQRT(XION)
          XIJ(1)=ICHARG*JCHARG*DUM1
          XIJ(2)=ICHARG**2*DUM1
          XIJ(3)=JCHARG**2*DUM1
          DO 1 I=1,3
            DUM=-1.2D-2*XIJ(I)**5.28D-1
            J1(I)=(4.D0+4.581D0*XIJ(I)**(-7.238D-1)*EXP(DUM)*(1.D0+
     >            7.238D-1+1.2D-2*5.28D-1*XIJ(I)**5.28D-1))/(4.D0+
     >            4.581D0*XIJ(I)**(-7.238D-1)*EXP(DUM))**2
1         CONTINUE
          EDFNCS=ICHARG*JCHARG/(8.D0*XION**2)*(XIJ(1)*J1(1)-5.D-1*
     >           XIJ(2)*J1(2)-5.D-1*XIJ(3)*J1(3))
     >           -ETHA/XION
       ENDIF
      END FUNCTION
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION PFUNC(THETA,ICHARG,JCHARG,A,XION)
       IMPLICIT REAL*8(A-H,O-Z)
C
       PFUNC=THETA+EFUNC(ICHARG,JCHARG,A,XION)
      END FUNCTION
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      END MODULE PitzH2SO4
