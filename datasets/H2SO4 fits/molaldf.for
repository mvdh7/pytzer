      PROGRAM MAIN
      IMPLICIT REAL*8(A-H,O-Z)
C--------------------------------------------------------------------
C     'accelerated' version - 8 july - needs testing. ok, 9 JULY.
C     working on this program! (10/11/93)
C--------------------------------------------------------------------
      PARAMETER(NV=1000,
     >          NCOEF=80,
     >          LIW=1,
     > LW=10+7*NCOEF+NCOEF*NCOEF+2*NV*NCOEF+3*NV+NCOEF*(NCOEF-1)/2)
C 
      INTEGER IW(LIW),NCELL(NV),NTYPE(NV),NABS(NV),SRC(NV),MKALPHA(NV),
     >        INDT(NV)
      REAL*8 MOLMAX,MOLMIN,W(LW),CJ(NCOEF),P(NCOEF),DELTA(NV),MPB(NV),
     >       MHG(NV),T(NV),HIGH(NV),LOW(NV),SUM(5),LOW2(NV),HIGH2(NV),
     >       CTOTAL2(NV)
C
!   ..user arrays (free to use for anything)
!     --------------------------------------
      INTEGER :: iUser(NV)
      REAL*8  :: User(NV)
C
      COMMON/USER1/CTOTAL(NV),Y(NV),FITTED(NV),WEIGHT(NV),SQRTW(NV),
     >             COEFF(NCOEF),T,MPB,MHG,EPS,LOW,HIGH,CTOTAL2,
     >             LOW2,HIGH2
      COMMON/USER1A/NHDIL,NCP,NEMF,NISO,NALPHA,MPAR(0:NCOEF+1),NCELL,
     >              NTYPE,INDT,SRC
C
      EXTERNAL LSFUN1
C
C
C
      MPAR(0)=-99
      MPAR(NCOEF+1)=-99
      EPS=X02AJF(1.D0)
C                   
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -                    
C  Setting Mark=0 reads phiL, any other number reads deltaHdil
C
      Mark=1
C
      If(Mark .EQ. 0) then
        OPEN(3,FILE='../data/hdilx.dat', STATUS='OLD')
      Else
        OPEN(3,FILE='../data/hdilxdf.dat', STATUS='OLD')
      Endif
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      OPEN(4,FILE='allmt.mst',STATUS='OLD')
      OPEN(1,FILE='../data/cpx2.dat',STATUS='OLD')
      OPEN(7,FILE='../data/emf.dat',STATUS='OLD')
      OPEN(8,FILE='../data/alpha.dat',STATUS='OLD')
      OPEN(9,FILE='../data/iso.dat',STATUS='OLD')
      OPEN(2,FILE='allmt.res',STATUS='UNKNOWN')
      OPEN(10,FILE='allmt_stats.res',STATUS='UNKNOWN')
C
C
C-----------------------------------------------------------------
C
10    READ(4,*,END=99,ERR=99) NHDIL,WTDIL, NCP,WTCP, NEMF,WTEMF,
     >                        NISO,WTISO, NALPHA,WTALPHA, MOLMIN,MOLMAX,
     >                        NPAR,
     >                        (MPAR(I),I=1,NPAR),
     >                        (P(I),I=1,NPAR),
     >                        (COEFF(I),I=1,NCOEF)
      NVAL=NHDIL+NCP+NEMF+NISO+NALPHA
C
      NHDIL_NonZeroWt = 0
      READ(3,*)
      IF(NHDIL.GT.0) THEN
      DO 1 I=1,NHDIL
        If(Mark .EQ. 0) then
          READ(3,*,END=99) CTOTAL(I), DUMH ,Y(I),WEIGHT(I),SRC(I)
          CTOTAL2(I)=0.D0
          T(I)=298.15D0
        Else
          READ(3,*,END=99) CTOTAL(I),CTOTAL2(I),Y(I),T(I),WEIGHT(I),
     >                     SRC(I)
          Y(I)=4.184D0 * Y(I)
        Endif
C
        IF(CTOTAL(I).GT.MOLMAX) WEIGHT(I)=0.D0
        IF(CTOTAL(I).LT.MOLMIN) WEIGHT(I)=0.D0
C
        IF(WEIGHT(I).GT.0.D0) THEN
          WEIGHT(I)=WEIGHT(I) * WTDIL
          NHDIL_NonZeroWt = NHDIL_NonZeroWt + 1
        ENDIF
C
        FITTED(I)=0.D0
        MPB(I)=0.D0
        MHG(I)=0.D0
        NTYPE(I)=4
        INDT(I)=-99
        SQRTW(I)=SQRT(WEIGHT(I))
1     CONTINUE
      REWIND 3
      ENDIF
C
      NCP_NonZeroWt = 0
      READ(1,*)
      IF(NCP.GT.0) THEN
      DO 11 I=NHDIL+1,NCP+NHDIL
        READ(1,*,END=99) CTOTAL(I), DUMH ,Y(I),WEIGHT(I),T(I),SRC(I)
        IF(CTOTAL(I).GT.MOLMAX) WEIGHT(I)=0.D0
        IF(CTOTAL(I).LT.MOLMIN) WEIGHT(I)=0.D0
C
        IF(WEIGHT(I).GT.1.D-6) THEN
          WEIGHT(I)=WEIGHT(I) * WTCP
          NCP_NonZeroWt = NCP_NonZeroWt + 1
        ENDIF
C
        FITTED(I)=0.D0
        MPB(I)=0.D0
        MHG(I)=0.D0
        CTOTAL2(I)=0.D0
        NTYPE(I)=5
        INDT(I)=-99
        SQRTW(I)=SQRT(WEIGHT(I))
11    CONTINUE
      REWIND 1
      ENDIF
C
      NEMF_NonZeroWt = 0
      READ(7,*)
      IF(NEMF.GT.0) THEN
      DO 51 I=NCP+NHDIL+1,NCP+NHDIL+NEMF
        READ(7,*,END=99) CTOTAL(I),Y(I),DE0,T(I),NCELL(I),
     >                   NABS(I),MPB(I),MHG(I),WEIGHT(I),SRC(I)      
        IF(CTOTAL(I).GT.MOLMAX) WEIGHT(I)=0.D0
        IF(CTOTAL(I).LT.MOLMIN) WEIGHT(I)=0.D0
C
        IF(WEIGHT(I).GT.1.D-6) THEN
          WEIGHT(I)=WEIGHT(I) * WTEMF
          NEMF_NonZeroWt = NEMF_NonZeroWt + 1 
        ENDIF
C
        IF(NABS(I).EQ.0) Y(I)=Y(I)*1.00033D0
        NTYPE(I)=3
        CTOTAL2(I)=0.D0
        IF(I.EQ.NCP+NHDIL+1) THEN
          INDT(I)=1
        ELSEIF(ABS(T(I)-T(I-1)).LT.1.D-6) THEN
          INDT(I)=-99
        ELSEIF(ABS(T(I)-T(I-1)).GT.1.D-6) THEN
          INDT(I)=1
        ENDIF
        IF(I.GT.NCP+NHDIL+1) THEN
          IF(INDT(I-1).EQ.1 .AND. WEIGHT(I-1).LT.1.D-7) INDT(I)=1
        ENDIF
        SQRTW(I)=SQRT(WEIGHT(I))
51    CONTINUE
      REWIND 7
      ENDIF
C
      NISO_NonZeroWt = 0
      READ(9,*)
      IF(NISO.GT.0) THEN
        DO 52 I=NCP+NHDIL+NEMF+1,NEMF+NCP+NHDIL+NISO
          READ(9,*,END=99) CTOTAL(I),Y(I),WEIGHT(I),T(I),SRC(I)      
          IF(CTOTAL(I).GT.MOLMAX) WEIGHT(I)=0.D0
          IF(CTOTAL(I).LT.MOLMIN) WEIGHT(I)=0.D0
C 
          IF(WEIGHT(I).GT.1.D-6) THEN
            WEIGHT(I)=WEIGHT(I) * WTISO
            NISO_NonZeroWt = NISO_NonZeroWt + 1
          ENDIF
C
          MPB(I)=0.D0
          MHG(I)=0.D0
          CTOTAL2(I)=0.D0
          NCELL(I)=0
          NTYPE(I)=1
          IF(I.EQ.NCP+NHDIL+NEMF+1) THEN
            INDT(I)=1
          ELSEIF(ABS(T(I)-T(I-1)).LT.1.D-6) THEN
            INDT(I)=-99
          ELSEIF(ABS(T(I)-T(I-1)).GT.1.D-6) THEN
            INDT(I)=1
          ENDIF
          IF(I.GT.NCP+NHDIL+NEMF+1) THEN
            IF(INDT(I-1).EQ.1 .AND. WEIGHT(I-1).LT.1.D-7) INDT(I)=1
          ENDIF
          SQRTW(I)=SQRT(WEIGHT(I))
52      CONTINUE
        REWIND 9
      ENDIF
C
      NALPHA_NonZeroWt = 0
      READ(8,*)
      IF(NALPHA.GT.0) THEN
        DO 53 I=NEMF+NCP+NHDIL+NISO+1,NVAL
          READ(8,*,END=99) CTOTAL(I),Y(I),MKALPHA(I),T(I),WEIGHT(I),
     >                     SRC(I)      
          IF(CTOTAL(I).GT.MOLMAX) WEIGHT(I)=0.D0
          IF(CTOTAL(I).LT.MOLMIN) WEIGHT(I)=0.D0
          IF(MKALPHA(I).EQ.2) WEIGHT(I)=0.D0
C
          IF(WEIGHT(I).GT.1.D-6) THEN
            WEIGHT(I)=WTALPHA*WEIGHT(I)
            NALPHA_NonZeroWt = NALPHA_NonZeroWt + 1
          ENDIF
C
          MPB(I)=0.D0
          MHG(I)=0.D0
          CTOTAL2(I)=0.D0
          NCELL(I)=0
          NTYPE(I)=2
          IF(I.EQ.NCP+NHDIL+NEMF+NISO+1) THEN
            INDT(I)=1
          ELSEIF(ABS(T(I)-T(I-1)).LT.1.D-6) THEN
            INDT(I)=-99
          ELSEIF(ABS(T(I)-T(I-1)).GT.1.D-6) THEN
            INDT(I)=1
          ENDIF
          IF(I.GT.NCP+NHDIL+NEMF+NISO+1) THEN
            IF(INDT(I-1).EQ.1 .AND. WEIGHT(I-1).LT.1.D-7) INDT(I)=1
          ENDIF
          SQRTW(I)=SQRT(WEIGHT(I))
53      CONTINUE
        REWIND 8
      ENDIF
C
      CLOSE(1)
      CLOSE(3)
      CLOSE(4)
      CLOSE(7)
      CLOSE(8)
      CLOSE(9)
C
C.... estimate H+ concentration bounds .............................
C
      DO 54 I=1,NVAL
C
C.....Condition:
C
       If((NCELL(I) .EQ. 2 .OR. NCELL(I).EQ.3) .AND. NTYPE(I).EQ.3) THEN
         CALL HG2SO4(CTOTAL(I),MHG(I),T(I))
         If(SQRT(ctotal(i)) .LE. 0.2D0) then
           weight(i)=0.5D0 * weight(i)
           sqrtw(i)=SQRT(weight(i))
         endif
       Endif
C
      IF(NTYPE(I).EQ.2 .AND. ABS(T(I)-298.15D0).GT.0.1D-3) THEN
        WEIGHT(I)=0.3D0 * WEIGHT(I)
        SQRTW(I)=SQRT(WEIGHT(I))
      ENDIF
C
C.....End of condition.
C
        HIGH(I)=2.D0*CTOTAL(I)
        LOW(I)=CTOTAL(I)-MPB(I)-MHG(I)
C
        IF(CTOTAL2(I) .GT. 0.D0) THEN
          HIGH2(I)=2.D0*CTOTAL2(I)
          LOW2(I)=CTOTAL2(I)
        ELSE 
          HIGH2(I)=0.D0
          LOW2(I)=0.D0
        ENDIF    
54    CONTINUE
C
C     **************************************************************
C     see notes approx 20 June '92 for equation for 1st dissociation
C     **************************************************************
      IFAIL = 1
      NCALL=1 
C
      CALL E04FYF(NVAL,NPAR,LSFUN1,P,FSUMSQ,W,LW,iUser,User,IFAIL)
      IF((IFAIL.NE.1).AND.(IFAIL.NE.2)) THEN
        WRITE(2,100) IFAIL
        WRITE(2,200) FSUMSQ
      ELSE
        WRITE(2,300) IFAIL
        STOP
      ENDIF
C
C     -----------------------------------------------------------
C     COMPUTE ESTIMATES OF THE VARIANCES OF THE SAMPLE REGRESSION
C     COEFFICIENTS AT THE FINAL POINT.
C     -----------------------------------------------------------
C     
C     E04YCF(JOB, M, N, FSUMSQ, S, V, LV, CJ, WORK, IFAIL 
C     
C     M = number of observations
C     
C     N = number of fitted parameters
C     
C     S(N) is the array of singular values of the Jacobian returned 
C          by E04FDF (in W, starting at W(NS)).
C     
C     V(LV,N) is the N x N right-hand orthogonal matrix of                  
C           J as returned by E04FDF. When V is passed in the
C           workspace array W (argument W(NV)) following E04FDF
C           LV must be the value N.
C     
C     CJ(N) when Job = 0, CJ returns the N diagonal elements of C.
C           That is to say CJ(1->N) contains the variances of 
C           fitted parameters 1 -> N.
C     WORK  is a work array (not used for anything special).
C     
C     ------------------------------------------------------
C     So, following E04FDF, the routine is called like this: 
C     
C     NS = 6*N + 2*M + M*N + 1 + MAX(1,(N*(N-1))/2)
C     NV = NS + N
C     IFAIL = 1
C     
C     CALL E04YCF(0,M,N,FSUMSQ,W(NS),W(NV),N,CJ,W,IFAIL)
C     
C     -------------------------------------------------------
C     
C     To experiment: try entering for 'M' in the argument list
C                    the number of non-zero weighted fitted points
C                    instead of the total.
C     
      NS = 6*NPAR + 2*NVAL + NVAL*NPAR + 1 + MAX(1,(NPAR*(NPAR-1))/2)
      NVX = NS + NPAR
C     
      nValNonZero = 0
      DO I=1,NVAL
        IF(WEIGHT(I) .GT. 1.D-7) nValNonZero = nValNonZero + 1
      ENDDO
C
C     
C   ..(new) compute the uncertainties
C     -----------------------------
      IFAIL = 1
      CALL E04YCF(0,nValNonZero,NPAR,FSUMSQ,W(NS),W(NVX),NPAR,CJ,W,
     >            IFAIL)
C
C
C   ..old code..
ccc   NS=6*NPAR+2*NVAL+NVAL*NPAR+1+MAX(1,(NPAR*(NPAR-1))/2)
ccc   NVX=NS+NPAR
ccc   IFAIL=1
ccc   CALL E04YCF(0,NVAL,NPAR,FSUMSQ,W(NS),W(NVX),NPAR,CJ,W,IFAIL)
C
      IF((IFAIL.NE.1).AND.(IFAIL.NE.2)) THEN
        WRITE(2,400) IFAIL
        WRITE(2,500) (J,P(J),SQRT(CJ(J)),P(J)/SQRT(CJ(J)),
     >                J=1,NPAR)
      ELSE
        WRITE(2,600) IFAIL
        STOP
      ENDIF
C
C
C     --------------------------------------
C     Numbers of non-zero-weighted points of 
C     each type.
C     --------------------------------------
      WRITE(10,660) NHDIL_NonZeroWt, NEMF_NonZeroWt, NCP_NonZeroWt,
     >              NISO_NonZeroWt, NALPHA_NonZeroWt
C
C
C     ------------------------------------------------------- 
C     Contributions to the weighted sum of squared deviations 
C     for each data type.
C     -------------------------------------------------------
      DO 3 I=1,NVAL
        IF(NTYPE(I).EQ.1) SUM(1)=SUM(1)+WEIGHT(I)*(Y(I)-FITTED(I))**2
        IF(NTYPE(I).EQ.2) SUM(2)=SUM(2)+WEIGHT(I)*(Y(I)-FITTED(I))**2
        IF(NTYPE(I).EQ.3) SUM(3)=SUM(3)+WEIGHT(I)*(Y(I)-FITTED(I))**2
        IF(NTYPE(I).EQ.4) SUM(4)=SUM(4)+WEIGHT(I)*(Y(I)-FITTED(I))**2
        IF(NTYPE(I).EQ.5) SUM(5)=SUM(5)+WEIGHT(I)*(Y(I)-FITTED(I))**2
3     CONTINUE
      WRITE(10,750) SUM(1),SUM(2),SUM(3),SUM(4),SUM(5)
C
C
C     -------------------------------------------------
C   ..Output the Npar "singular values of the Jacobean"
C     -------------------------------------------------
cc      WRITE(10,670)
cc      DO I=1,NPar
C
cc        WRITE(10,675) I, W(NS-1 + I)
C
cc      ENDDO
C
C
C     ----------------------------------------
C   ..Compute and output the covariance matrix
C     ----------------------------------------
      iFail = 1
      Call E04YCF(-1, nValNonZero, nPar, FSumSQ, W(NS), W(NVX), nPar, 
     >            CJ, W, iFail)
C
      WRITE(10,645)  ! header
      DO I=NVX,NVX+nPar-1
C
        WRITE(10,646) (W(I + J*nPar), J=0,nPar-1)
C
      ENDDO
C
C     ----------------------------------------
C   ..Output all measured and fitted values
C     ----------------------------------------
      WRITE(2,650)
      DO 2, J=1,NVAL
        IF(WEIGHT(J).GT.0.D0) THEN
          DELTA(J)=FITTED(J)-Y(J)
        ELSE
          DELTA(J)=0.D0
        ENDIF
        WRITE(2,700) J,NTYPE(J),SRC(J),CTOTAL(J),Y(J),FITTED(J),
     >               DELTA(J),T(J),WEIGHT(J),NCELL(J)
2     CONTINUE
C
99    CONTINUE
      STOP
C
100   FORMAT(//' EXIT OK FROM E04FDF: IFAIL = ',I3)
200   FORMAT(1X,'ON EXIT, SUM OF SQUARES = ',F12.6)
300   FORMAT(/' !ERROR ON EXIT, NAG FAILURE: IFAIL= ',I1,/
     >        ' *** TERMINATED ***')
400   FORMAT(/' EXIT OK FROM E04YCF: IFAIL = ',I3)
500   FORMAT(/' PARAMETERS AND THEIR STANDARD ERRORS:',/
     >        T8,'PARAMETER',T23,'STD. ERROR',T36,'ERR. RATIO'/
     >        (1X,T2,I2,T6,E17.9,T24,E12.4,T37,F9.4))
600   FORMAT(//' !ERROR EXIT FROM E04YCF:  IFAIL = ',I3,/
     >         ' - SEE ROUTINE DOCUMENT')
650   FORMAT(///' ------------------- RESULTS ---------------------',//
     > 2X,'I',T7,'T',2X,'SRC',T16,'mTOT',T31,'Y VALUE',T45,'FITTED',
     >  T61,'DELTA ')
700   FORMAT(2X,I3,2X,I1,2X,I3,T15,E13.5,T31,E13.5,T46,E13.5,T61,F12.6,
     >       2X,E13.5,2X,E10.3,2X,I3)
750   FORMAT(//1X,'SQ OSM  = ',E10.4,3X,'SQ ALPHA = ',E10.4,3X,
     >          'SQ EMF = ',E10.4,/1X,'SQ HDIL = ',E10.4,3X,
     >          'SQ PHICP = ',E10.4)
C
C   ..added formats
C     -------------
645   FORMAT(//1X,' Covariance Matrix:'/)
646   FORMAT(1X,100(E11.4,2X))
660   FORMAT(//1X,'Numbers of non-zero-weighted points of each type'
     >//1X, '{delta}Hdil = ',I4/1X, 'EMF = ',I4/1X, 'Cp = ',I4/1X,
     >'Isopiestic = ',I4/1X, 'Alpha = ',I4)
ccc 670   FORMAT(//1X,'Singular values of the Jacobean')
ccc 675   FORMAT(1X,I2,3X,E11.4)
C
      END
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE LSFUN1(NVAL,NPAR,P,RESID,iUser,User)
      IMPLICIT REAL*8 (A-H,O-Z)
C
      PARAMETER(NV=1000,
     >          NCOEF=80)
C
      INTEGER NCELL(NV),NTYPE(NV),INDT(NV),SRC(NV),iUser(*)
      REAL*8 P(NPAR),RESID(NVAL)
      REAL*8 MPB(NV),MHG(NV),T(NV),HIGH(NV),LOW(NV),HIGH2(NV),LOW2(NV),
     >       CTOTAL2(NV),User(*)
C
      COMMON/USER1/CTOTAL(NV),Y(NV),FITTED(NV),WEIGHT(NV),SQRTW(NV),
     >             COEFF(NCOEF),T,MPB,MHG,EPS,LOW,HIGH,CTOTAL2,
     >             LOW2,HIGH2
      COMMON/USER1A/NHDIL,NCP,NEMF,NISO,NALPHA,MPAR(0:NCOEF+1),NCELL,
     >              NTYPE,INDT,SRC
C
      ID=1
C
C--------------------------------------------------------     
C
      DO 1 I=1,NPAR
        IF(MPAR(I).NE.MPAR(I+1) .AND. MPAR(I).NE.MPAR(I-1)) THEN
          COEFF(MPAR(I))=P(I)
        ENDIF
1     CONTINUE
C
c      WRITE(*,'(1X,''P='',(4(E18.10,1X)))') (P(J),J=1,NPAR)
C     WRITE(*,'(1X,''C = '',(10E11.4,1X))') (COEFF(J),J=1,NPAR)
C
C---------------------------------------------------------
C
C
      IF(NEMF.GT.0)
     >CALL EMF(T,COEFF,LOW,HIGH,SRC,
     >         CTOTAL,FITTED,WEIGHT,INDT,MHG,MPB,NCELL,
     >         NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
C
      IF(NALPHA.GT.0)
     >CALL ALPHAC(T,COEFF,LOW,HIGH,
     >            CTOTAL,FITTED,WEIGHT,INDT,
     >            NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
C
      IF(NISO.GT.0)
     >CALL ISOC(T,COEFF,LOW,HIGH,CTOTAL,FITTED,WEIGHT,INDT,
     >          NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
C
      IF(NHDIL.GT.0)
     >CALL PHIL(T,COEFF,LOW,HIGH,CTOTAL,LOW2,HIGH2,CTOTAL2,FITTED,
     >          WEIGHT,INDT,NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
C
      IF(NCP.GT.0) 
     >CALL PHICP(T,COEFF,LOW,HIGH,CTOTAL,FITTED,WEIGHT,INDT,
     >           NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
C
      DO 2 I=1,NVAL
       RESID(I)=SQRTW(I)*(Y(I)-FITTED(I))
C
C--------------------------------------------------------------------
c       WRITE(2,'(1X,I2,2X,6(D18.10,2X))') NTYPE(I),CTOTAL(I),Y(I),
c     >        FITTED(I),T(I),WEIGHT(I)
C--------------------------------------------------------------------
2     CONTINUE
C
c      IF(ID.EQ.1) STOP
C
      RETURN
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE HG2SO4(M,MHG,TABS)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 M,MHG
C
C     this subroutine calculates Hg2SO4 solubilities from those listed
C     for 0 and 28oC in Silcock, for cells II and III. Note conc. limits
C     below (in fact there are no data below 0.001 mH2SO4).
C
        IF(M.LT.0.001D0 .OR. M.GT.3.41D0) THEN
          MHG=0.D0
        ELSE
          T=TABS-273.15D0

C---------- incorrect Silcock based values ----------------------------
c          CONC0=1.84181D-4 + 4.10828D-4*SQRT(M) - 1.39554D-4*M**2 +
c     >          4.5596D-5*M**2.5D0 - 5.28195D-8/M + 1.78466D-5/SQRT(M)
c          CONC28=2.45972D-4 + 1.26155D-3*SQRT(M) - 5.64492D-4*M -
c     >           2.93633D-7/M + 3.45118D-5/SQRT(M)
C
C---------- from table 5 of Craig et al paper -------------------------
           CONC0=2.3141D-4 + 5.05109D-4*SQRT(M) - 1.71473D-4*M**2 +
     >           5.60029D-5*M**2.5D0 - 5.38811D-8/M + 2.17084D-5/SQRT(M)
           CONC28=3.34605D-4 + 1.42062D-3*SQRT(M) - 5.07682D-4*M -
     >           3.36919D-7/M + 4.08982D-5/SQRT(M) - 1.1684D-4*M**2 +
     >           4.40779D-5*M**2.5D0
C
           SLOPE=(CONC28-CONC0)/28.D0
           MHG=CONC0 + T*SLOPE
        ENDIF
C
        RETURN
        END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE PHIL(T,COEFF,LOW,HIGH,CTOTAL,LOW2,HIGH2,CTOTAL2,
     >                FITTED,WEIGHT,INDT,
     >                NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER(NV=1000,
     >          NCOEF=80,
     >          HREL=5.D-3)
C
      INTEGER INDT(NV)
      REAL*8 T(NV),CTOTAL(NV),FITTED(NV),WEIGHT(NV),COEFF(NCOEF),
     >       HIGH(NV),LOW(NV),LOW2(NV),HIGH2(NV),CTOTAL2(NV)
      EXTERNAL FC05
C
      DO 1 I=1,NHDIL
        IF(WEIGHT(I).GT.0.D0) THEN
          IF(CTOTAL2(I).EQ.0.D0) THEN
C
C
          CALL CP(T(I),HREL,CTOTAL(I),LOW(I),HIGH(I),COEFF,
     >            DIFF1,DUM,EPS)
          FITTED(I)= - 8.3144D0/CTOTAL(I)*T(I)**2*DIFF1
C
C
          ELSE
C
C
          CALL CP(T(I),HREL,CTOTAL(I),LOW(I),HIGH(I),COEFF,
     >            DIFF1,DUM,EPS)
          CALL CP(T(I),HREL,CTOTAL2(I),LOW2(I),HIGH2(I),COEFF,
     >            DIFF2,DUM,EPS)
C
          FITTED(I)= 8.3144D0*T(I)**2
     >               * (DIFF1/CTOTAL(I) - DIFF2/CTOTAL2(I))
          ENDIF
        ELSE
          FITTED(I)=0.D0
        ENDIF
1     CONTINUE
      RETURN
      END 
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE PHICP(T,COEFF,LOW,HIGH,
     >                CTOTAL,FITTED,WEIGHT,INDT,
     >                NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER(NV=1000,
     >          NCOEF=80,
     >          HREL=5.D-3)
C
      INTEGER INDT(NV)
      REAL*8 T(NV),CTOTAL(NV),FITTED(NV),WEIGHT(NV),COEFF(NCOEF),
     >       HIGH(NV),LOW(NV)
      EXTERNAL FC05
C
      DO 1 I=NHDIL+1,NHDIL+NCP
        IF(WEIGHT(I).GT.1.D-7) THEN
          TEMP=T(I)
          CALL CP(T(I),HREL,CTOTAL(I),LOW(I),HIGH(I),COEFF,
     >            DIFF1,DIFF2,EPS)
          DELTA=TEMP-298.15D0
          CPZERO=COEFF(28)*1.D1 + DELTA*COEFF(48) 
     >                          + DELTA**2*COEFF(49)*1.D-1
          FITTED(I)=CPZERO - (8.3144D0/CTOTAL(I))
     >              *(2.D0*TEMP*DIFF1 + TEMP**2*DIFF2) 
        ELSE
          FITTED(I)=0.D0
        ENDIF
1     CONTINUE
      RETURN
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE EMF(T,COEFF,LOW,HIGH,SRC,
     >               CTOTAL,FITTED,WEIGHT,INDT,MHG,MPB,NCELL,
     >               NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER(NV=1000,NCOEF=80)
C
      INTEGER INDT(NV),NCELL(NV),SRC(NV)
      REAL*8 MH,MHSO4,MSO4
      REAL*8 T(NV),CTOTAL(NV),FITTED(NV),WEIGHT(NV),COEFF(NCOEF),
     >       B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),
     >       PSIC(5,5,5),PSIA(5,5,5),MHG(NV),MPB(NV),HIGH(NV),LOW(NV) 
      EXTERNAL FC05
C....................................................................
C
C    for R=8.3144 and F=96484.6, R/F = 8.61733375D-5. This now appears
C    in the 'E = ' equations below.
C
C....................................................................
C
      DO 1 I=NHDIL+NCP+1,NHDIL+NCP+NEMF
        IF(WEIGHT(I).GT.1.D-7) THEN
          TP=T(I)
          IF(INDT(I).EQ.1) 
     >      CALL PARASET(TP,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
          MH=ZBRENT(FC05,LOW(I),HIGH(I),TP,CTOTAL(I),COEFF, 
     >              MHG(I),MPB(I),EPS,
     >              B,C,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
          MHSO4=CTOTAL(I)*2.D0 - MH
          MSO4=CTOTAL(I)+MPB(I)+MHG(I)-MHSO4
C....................................................................
C
C delta E0 for src = 33 at 298.15K = coeff(58) (Cell 1)
C delta E0 for src = 34 at 298.15K = coeff(59) (Cell 2)
C delta E0 for src = 37 at 298.15K = coeff(60) (Cell 3)
C delta E0 for CELL= 2  at 318.15K = coeff(61)          
C delta E0 for CELL= 4  at 285.65K = coeff(62)            
C
          TTEST=ABS(TP-298.15D0)
          IF(NCELL(I).EQ.1) THEN
            EZERO1=COEFF(16)*10.D0*(1.D0/TP-1.D0/298.15D0)
     >      +COEFF(17)*1.D-3*(TP*LOG(TP)-1698.7384D0) + COEFF(18)
            IF(SRC(I).EQ.33 .AND. TTEST.LE.1.D-6) THEN
              EZERO1=EZERO1 + COEFF(58)
            ENDIF
            AW=EXP(AWLG)
            FITTED(I)= EZERO1 +(0.5D0*8.61733375D-5*TP)
     >                *LOG(ACTH**2*ACTSO4*MH**2*MSO4/AW**2)
          ELSEIF(NCELL(I).EQ.2) THEN
            EZERO2=COEFF(19)*10.D0*(1.D0/TP-1.D0/298.15D0)
     >      +COEFF(20)*1.D-3*(TP*LOG(TP)-1698.7384D0) + COEFF(21)  
            IF(SRC(I).EQ.34 .AND. TTEST.LE.1.D-6) THEN
              EZERO2=EZERO2 + COEFF(59)
            ELSEIF(SRC(I).EQ.37 .AND. TTEST.LE.1.D-6) THEN 
              EZERO2=EZERO2 + COEFF(60)   
            ENDIF
            IF(ABS(TP-318.15D0) .LE. 1.D-6) EZERO2=EZERO2 + COEFF(61)   
            FITTED(I)= EZERO2 -(0.5D0*8.61733375D-5*TP)
     >                *LOG(ACTH**2*ACTSO4*MH**2*MSO4)
          ELSEIF(NCELL(I).EQ.3) THEN
            EZERO3=COEFF(15)
            AW=EXP(AWLG)
            FITTED(I)= EZERO3+(8.61733375D-5*TP)
     >                 *LOG(ACTH**2*ACTSO4*MH**2*MSO4/AW)
          ELSEIF(NCELL(I).EQ.4) THEN
            EZERO4=COEFF(22)*10.D0*(1.D0/TP-1.D0/298.15D0)
     >      +COEFF(23)*1.D-3*(TP*LOG(TP)-1698.7384D0) + COEFF(24) 
            IF(ABS(TP-285.65D0) .LE. 1.D-6) EZERO4 = EZERO4 + COEFF(62)
            FITTED(I)= EZERO4 +(0.5D0*8.61733375D-5*TP)
     >                *LOG(ACTH**2*ACTSO4*MH**2*MSO4)
          ENDIF
C
        ELSE
          FITTED(I)=0.D0
        ENDIF
1     CONTINUE
      RETURN
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
C THIS BLANKED OFF ROUTINE IS FOR ESTABLISHING EACH EZERO INDIVIDUALLY
C
c     SUBROUTINE EMF(T,COEFF,LOW,HIGH,SRC,
c    >               CTOTAL,FITTED,WEIGHT,INDT,MHG,MPB,NCELL,
c    >               NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
c     IMPLICIT REAL*8 (A-H,O-Z)
c     PARAMETER(NV=1000,
c    >          NCOEF=80)
C
c     INTEGER INDT(NV),NCELL(NV),SRC(NV)
c     REAL*8 MH,MHSO4,MSO4
c     REAL*8 T(NV),CTOTAL(NV),FITTED(NV),WEIGHT(NV),COEFF(NCOEF),
c    >       B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),
c    >       PSIC(5,5,5),PSIA(5,5,5),MHG(NV),MPB(NV),HIGH(NV),LOW(NV)
c     EXTERNAL FC05
C....................................................................
C
C    for R=8.3144 and F=96484.6, R/F = 8.61733375D-5. This now appears
C    in the 'E = ' equations below.
C
C....................................................................
C
c     DO 1 I=NHDIL+NCP+1,NHDIL+NCP+NEMF
c       IF(WEIGHT(I).GT.1.D-7) THEN
c         TP=T(I)
c         IF(INDT(I).EQ.1) 
c    >      CALL PARASET(TP,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
c         MH=ZBRENT(FC05,LOW(I),HIGH(I),TP,CTOTAL(I),COEFF,
c    >              MHG(I),MPB(I),EPS,
c    >              B,C,THETAC,THETAA,PSIC,PSIA,
c    >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
c         MHSO4=CTOTAL(I)*2.D0 - MH
c         MSO4=CTOTAL(I)+MPB(I)+MHG(I)-MHSO4
C....................................................................
C
C delta E0 for src = 33 at 298.15K = coeff(58) (Cell 1)
C delta E0 for src = 34 at 298.15K = coeff(59) (Cell 2)
C delta E0 for src = 37 at 298.15K = coeff(60) (Cell 2)
C
c         TTEST=ABS(TP-298.15D0)
c         IF(NCELL(I).EQ.1) THEN
c           IF(ABS(TP-298.15D0) .LE. 1.D-6) EZERO1 = COEFF(17)
c           IF(ABS(TP-278.15D0) .LE. 1.D-6) EZERO1 = COEFF(18)
c           IF(ABS(TP-283.15D0) .LE. 1.D-6) EZERO1 = COEFF(19)
c           IF(ABS(TP-293.15D0) .LE. 1.D-6) EZERO1 = COEFF(20)
c           IF(ABS(TP-308.15D0) .LE. 1.D-6) EZERO1 = COEFF(21)
c           IF(ABS(TP-318.15D0) .LE. 1.D-6) EZERO1 = COEFF(22)
c           IF(ABS(TP-328.15D0) .LE. 1.D-6) EZERO1 = COEFF(23)
C
c           IF(SRC(I).EQ.33 .AND. TTEST.LE.1.D-6) THEN
c             EZERO1=EZERO1 + COEFF(58)
c           ENDIF
c           AW=EXP(AWLG)
c           FITTED(I)= EZERO1 +(0.5D0*8.61733375D-5*TP)
c    >                *LOG(ACTH**2*ACTSO4*MH**2*MSO4/AW**2)
c         ELSEIF(NCELL(I).EQ.2) THEN
c           IF(ABS(TP-298.15D0) .LE. 1.D-6) EZERO2 = COEFF(62)
c           IF(ABS(TP-278.15D0) .LE. 1.D-6) EZERO2 = COEFF(63)
c           IF(ABS(TP-288.15D0) .LE. 1.D-6) EZERO2 = COEFF(64)
c           IF(ABS(TP-308.15D0) .LE. 1.D-6) EZERO2 = COEFF(65)
c           IF(ABS(TP-318.15D0) .LE. 1.D-6) EZERO2 = COEFF(66)
c           IF(ABS(TP-328.15D0) .LE. 1.D-6) EZERO2 = COEFF(67)
C
c           IF(SRC(I).EQ.34 .AND. TTEST.LE.1.D-6) THEN
c             EZERO2=EZERO2 + COEFF(59)
c           ELSEIF(SRC(I).EQ.37 .AND. TTEST.LE.1.D-6) THEN 
c             EZERO2=EZERO2 + COEFF(60)   
c           ENDIF
c           FITTED(I)= EZERO2 -(0.5D0*8.61733375D-5*TP)
c    >                *LOG(ACTH**2*ACTSO4*MH**2*MSO4)
c         ELSEIF(NCELL(I).EQ.3) THEN
c           EZERO3=COEFF(15)
c           AW=EXP(AWLG)
c           FITTED(I)= EZERO3+(8.61733375D-5*TP)
c    >                 *LOG(ACTH**2*ACTSO4*MH**2*MSO4/AW)
c         ELSEIF(NCELL(I).EQ.4) THEN
c           IF(ABS(TP-273.15D0) .LE. 1.D-6) EZERO4 = COEFF(68)
c           IF(ABS(TP-285.65D0) .LE. 1.D-6) EZERO4 = COEFF(69)
c           IF(ABS(TP-310.65D0) .LE. 1.D-6) EZERO4 = COEFF(70)
c           IF(ABS(TP-323.15D0) .LE. 1.D-6) EZERO4 = COEFF(71)
c           IF(ABS(TP-298.15D0) .LE. 1.D-6) EZERO4 = COEFF(72)
C
c           FITTED(I)= EZERO4 +(0.5D0*8.61733375D-5*TP)
c    >                *LOG(ACTH**2*ACTSO4*MH**2*MSO4)
c         ENDIF
C
c       ELSE
c         FITTED(I)=0.D0
c       ENDIF
c1      CONTINUE
c     RETURN
c     END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE ISOC(T,COEFF,LOW,HIGH,
     >                CTOTAL,FITTED,WEIGHT,INDT,
     >                NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER(NV=1000,
     >          NCOEF=80)
C
      INTEGER INDT(NV)
      REAL*8 MH,MHSO4,MSO4
      REAL*8 T(NV),CTOTAL(NV),FITTED(NV),WEIGHT(NV),COEFF(NCOEF),
     >       HIGH(NV),LOW(NV),B(5,5,3),C(5,5),THETAC(5,5),
     >       THETAA(5,5),PSIC(5,5,5),PSIA(5,5,5)
      EXTERNAL FC05
C
      DO 1 I=NHDIL+NCP+NEMF+1,NHDIL+NCP+NEMF+NISO
        IF(WEIGHT(I).GT.1.D-7) THEN
          IF(INDT(I).EQ.1) 
     >       CALL PARASET(T(I),COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
          MH=ZBRENT(FC05,LOW(I),HIGH(I),T(I),CTOTAL(I),COEFF,
     >              0.D0,0.D0,EPS,
     >              B,C,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
          MHSO4=CTOTAL(I)*2.D0 - MH
          MSO4=CTOTAL(I)-MHSO4
          FITTED(I)=-AWLG/(0.0180152D0*3.D0*CTOTAL(I))
        ELSE
          FITTED(I)=0.D0
        ENDIF
1     CONTINUE
      RETURN
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      SUBROUTINE ALPHAC(T,COEFF,LOW,HIGH,
     >                  CTOTAL,FITTED,WEIGHT,INDT,
     >                  NHDIL,NCP,NEMF,NISO,NALPHA,EPS)
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER(NV=1000,
     >          NCOEF=80)
C
      INTEGER INDT(NV)
      REAL*8 MH,MHSO4,MSO4
      REAL*8 T(NV),CTOTAL(NV),FITTED(NV),WEIGHT(NV),COEFF(NCOEF),
     >       B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),
     >       PSIC(5,5,5),PSIA(5,5,5),HIGH(NV),LOW(NV)
      EXTERNAL FC05
C
      DO 1 I=NHDIL+NCP+NEMF+NISO+1,NHDIL+NCP+NEMF+NISO+NALPHA
        IF(WEIGHT(I).GT.1.D-7) THEN
          IF(INDT(I).EQ.1) 
     >       CALL PARASET(T(I),COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
          MH=ZBRENT(FC05,LOW(I),HIGH(I),T(I),CTOTAL(I),COEFF,
     >              0.D0,0.D0,EPS,
     >              B,C,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
          MHSO4=CTOTAL(I)*2.D0 - MH
          MSO4=CTOTAL(I)-MHSO4
          FITTED(I)=MSO4/(MSO4+MHSO4)
        ELSE
          FITTED(I)=0.D0
        ENDIF
1     CONTINUE
      RETURN
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION FC05(MH,TEMP,MTOT,COEFF,MHG1,MPB1,CONST,APHI,
     >              B,C,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      IMPLICIT REAL*8 (A-H,O-Z)
      PARAMETER(NV=1000,
     >          NCOEF=80,
     >          R=8.3144D0)
C
      REAL*8 MTOT,MH,MHSO4,MSO4,KSTAR,MHG1,MPB1
      REAL*8 CATM(5),ANM(5),NEUTM(5),COEFF(NCOEF),
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
     >         ACTH,ACTHSO4,ACTSO4,OSM,AWLG,TEMP,COEFF(50))
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
      END
C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION AFT(T)
      IMPLICIT REAL*8 (A-H,O-Z)
      REAL*8 AA(19)
C
      EXTERNAL E02AKF
      DATA AA/0.797256081240D+00,0.573389669896D-01,
     > 0.977632177788D-03, 0.489973732417D-02,-0.313151784342D-02,
     > 0.179145971002D-02,-0.920584241844D-03, 0.443862726879D-03,
     >-0.203661129991D-03, 0.900924147948D-04,-0.388189392385D-04,
     > 0.164245088592D-04,-0.686031972567D-05, 0.283455806377D-05,
     >-0.115641433004D-05, 0.461489672579D-06,-0.177069754948D-06,
     > 0.612464488231D-07,-0.175689013085D-07/
     > XMIN,XMAX/0.234150D+03,0.373150D+03/ 
C
       IFAIL=0
       CALL E02AKF(19,XMIN,XMAX,AA,1,19,T,AFT,IFAIL)
C
       IF(IFAIL.NE.0) THEN
         WRITE(2,'(1X,''IFAIL = '',I2,'' STOP'')') IFAIL
         STOP
       ENDIF
       END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION GFUNC(ALPHA,XION)
       IMPLICIT REAL*8(A-H,O-Z)
       DUM=ALPHA*SQRT(XION)
       GFUNC=2.D0*(1.D0-(1.D0+DUM)*EXP(-DUM))/DUM**2
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION GDFUNC(ALPHA,XION)
       IMPLICIT REAL*8(A-H,O-Z)
       DUM=ALPHA*SQRT(XION)
       GDFUNC=-2.D0*(1.D0-(1.D0+DUM+5.D-1*DUM**2)*EXP(-DUM))/DUM**2
      END
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
      END
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
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION PFUNC(THETA,ICHARG,JCHARG,A,XION)
       IMPLICIT REAL*8(A-H,O-Z)
C
       PFUNC=THETA+EFUNC(ICHARG,JCHARG,A,XION)
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION ZBRENT(FC05,LOW,HIGH,TEMP,MTOT,COEFF,MHG1,MPB1,EPS,
     >                BCOEF,CCOEF,THETAC,THETAA,PSIC,PSIA,
     >                ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      IMPLICIT REAL*8(A-H,O-Z)
C
      PARAMETER(ITMAX=50,TOL=1.D-16,
     >          NCOEF=80,TR=298.15D0,RG=8.3144D0,P1=0.0128310991D0,
     >          P2=-9.486301516D0,P3=1962.6177D0)
C
      REAL*8 MTOT,MHG1,MPB1,COEFF(NCOEF),BCOEF(5,5,3),CCOEF(5,5),
     >       THETAC(5,5),THETAA(5,5),PSIC(5,5,5),PSIA(5,5,5),LOW
C
C------------------------------------------------------------
C     !subroutine contains several 'IF's where real*8 variable
C     is tested for being = 0.D0
C
C------------------------------------------------------------
C
C.......below follows equation (for DISSociation) based on Dickson's
C       eq 6, but yielding correct 298K value. Note CONST = ...
C
       DUM=562.694864456D0 - 102.5154D0*LOG(TEMP) - 1.117033D-4*TEMP**2
     >     + 0.2477538D0*TEMP - 13273.75D0/TEMP
       CONST=10.D0**(-DUM)
       APHI=AFT(TEMP)
C...................................................................
C
c.      IF(ABS(TEMP-298.15D0) .LE. 1.D-6) THEN
c.        CONST=95.238095D0
c.        APHI=0.3914752375D0
c.      ELSE
c.        DUM=LOG(COEFF(25))+
c.     >      (COEFF(26)*1.D2/RG)*(1.D0/TR-1.D0/TEMP)+
c.     >      P3/RG*(TR/TEMP - (1.D0+LOG(TR/TEMP)))+ 
c.     >      P1/(3.D0*RG)*(0.5D0*(TEMP**2-TR**2)+TR**2*(TR/TEMP-1.D0))+
c.     >      P2/(2.D0*RG)*((TEMP-TR)+TR*(TR/TEMP-1.D0))
c.        CONST=EXP(DUM)
c.        APHI=AFT(TEMP)
c.      ENDIF
C
      A=LOW
      B=HIGH
C
      FA=FC05(A,TEMP,MTOT,COEFF,MHG1,MPB1,CONST,APHI,
     >              BCOEF,CCOEF,THETAC,THETAA,PSIC,PSIA,
     >              ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      FB=FC05(B,TEMP,MTOT,COEFF,MHG1,MPB1,CONST,APHI,
     >              BCOEF,CCOEF,THETAC,THETAA,PSIC,PSIA,
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
        FB=FC05(B,TEMP,MTOT,COEFF,MHG1,MPB1,CONST,APHI,
     >              BCOEF,CCOEF,THETAC,THETAA,PSIC,PSIA,
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
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      SUBROUTINE CP(TDATA,HREL,MTOT,LOW,HIGH,COEFF,DIFF1,DIFF2,EPS)
      IMPLICIT REAL*8 (A-H,O-Z)
C
      PARAMETER(NCOEF=80)       
C
      REAL*8 MTOT,MH,MHSO4,MSO4,COEFF(NCOEF),LOW,HIGH
      REAL*8 B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),
     >       PSIC(5,5,5),PSIA(5,5,5)
C
      EXTERNAL FC05
C
      H=TDATA*HREL
C     WRITE(*,'(1X,4(E11.4,2X))') MTOT,AA,BB
      DF1A=0.D0
      DF1B=0.D0
      DF2=0.D0
C
      T=TDATA
      CALL PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
      MH=ZBRENT(FC05,LOW,HIGH,T,MTOT,COEFF,0.D0,0.D0,EPS,
     >          B,C,THETAC,THETAA,PSIC,PSIA,
     >          ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      MHSO4=MTOT*2.D0 - MH
      MSO4=MTOT-MHSO4
      GSRTVAL=GSRT(MH,MHSO4,MSO4,MTOT,ACTH,ACTHSO4,ACTSO4,OSM)
C     WRITE(*,'(1X,''GSRTH'',12(E13.6,1X))') T,MTOT,MH,MHSO4,MSO4,ACTH,
C    >      ACTHSO4,ACTSO4,OSM,GSRTVAL,EXP2
      DF2=DF2 - 30.D0*GSRTVAL
C
      T=TDATA+2.D0*H
      CALL PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
      MH=ZBRENT(FC05,LOW,HIGH,T,MTOT,COEFF,0.D0,0.D0,EPS,
     >          B,C,THETAC,THETAA,PSIC,PSIA,
     >          ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      MHSO4=MTOT*2.D0 - MH
      MSO4=MTOT-MHSO4
      GSRTVAL=GSRT(MH,MHSO4,MSO4,MTOT,ACTH,ACTHSO4,ACTSO4,OSM)
C     WRITE(*,'(1X,''GSRTH'',12(E13.6,1X))') T,MTOT,MH,MHSO4,MSO4,ACTH,
C    >      ACTHSO4,ACTSO4,OSM,GSRTVAL,EXP2
      DF1A=DF1A - GSRTVAL
      DF2=DF2 - GSRTVAL
C 
      T=TDATA+H
      CALL PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
      MH=ZBRENT(FC05,LOW,HIGH,T,MTOT,COEFF,0.D0,0.D0,EPS,
     >          B,C,THETAC,THETAA,PSIC,PSIA,
     >          ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      MHSO4=MTOT*2.D0 - MH                     
      MSO4=MTOT-MHSO4                          
      GSRTVAL=GSRT(MH,MHSO4,MSO4,MTOT,ACTH,ACTHSO4,ACTSO4,OSM)
C     WRITE(*,'(1X,''GSRTH'',12(E13.6,1X))') T,MTOT,MH,MHSO4,MSO4,ACTH,
C    >      ACTHSO4,ACTSO4,OSM,GSRTVAL,EXP1
      DF1A=DF1A + 8.D0*GSRTVAL                   
      DF1B=DF1B - GSRTVAL                        
      DF2=DF2 + 16.D0*GSRTVAL
C
      T=TDATA-H
      CALL PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
      MH=ZBRENT(FC05,LOW,HIGH,T,MTOT,COEFF,0.D0,0.D0,EPS,
     >          B,C,THETAC,THETAA,PSIC,PSIA,
     >          ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      MHSO4=MTOT*2.D0 - MH
      MSO4=MTOT-MHSO4 
      GSRTVAL=GSRT(MH,MHSO4,MSO4,MTOT,ACTH,ACTHSO4,ACTSO4,OSM)
C     WRITE(*,'(1X,''GSRTH'',12(E13.6,1X))') T,MTOT,MH,MHSO4,MSO4,ACTH,
C    >      ACTHSO4,ACTSO4,OSM,GSRTVAL,EXM1
      DF1A=DF1A - 8.D0*GSRTVAL
      DF1B=DF1B + GSRTVAL
      DF2=DF2 + 16.D0*GSRTVAL
C
      T=TDATA-2.D0*H
      CALL PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
      MH=ZBRENT(FC05,LOW,HIGH,T,MTOT,COEFF,0.D0,0.D0,EPS,
     >          B,C,THETAC,THETAA,PSIC,PSIA,
     >          ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      MHSO4=MTOT*2.D0 - MH
      MSO4=MTOT-MHSO4
      GSRTVAL=GSRT(MH,MHSO4,MSO4,MTOT,ACTH,ACTHSO4,ACTSO4,OSM)
C     WRITE(*,'(1X,''GSRTH'',13(E13.6,1X))') T,MTOT,MH,MHSO4,MSO4,ACTH,
C    >      ACTHSO4,ACTSO4,OSM,GSRTVAL,EXM2,DIFFZ*1.766E5/MTOT
      DF1A=DF1A + GSRTVAL
      DF2=DF2 - GSRTVAL
C
      T=TDATA+H/2.D0
      CALL PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
      MH=ZBRENT(FC05,LOW,HIGH,T,MTOT,COEFF,0.D0,0.D0,EPS,
     >          B,C,THETAC,THETAA,PSIC,PSIA,
     >          ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      MHSO4=MTOT*2.D0 - MH
      MSO4=MTOT-MHSO4 
      GSRTVAL=GSRT(MH,MHSO4,MSO4,MTOT,ACTH,ACTHSO4,ACTSO4,OSM)
      DF1B=DF1B + 8.D0*GSRTVAL
C
      T=TDATA-H/2.D0
      CALL PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
      MH=ZBRENT(FC05,LOW,HIGH,T,MTOT,COEFF,0.D0,0.D0,EPS,
     >          B,C,THETAC,THETAA,PSIC,PSIA,
     >          ACTH,ACTHSO4,ACTSO4,OSM,AWLG)
      MHSO4=MTOT*2.D0 - MH
      MSO4=MTOT-MHSO4
      GSRTVAL=GSRT(MH,MHSO4,MSO4,MTOT,ACTH,ACTHSO4,ACTSO4,OSM)
      DF1B=DF1B - 8.D0*GSRTVAL
C
      DF1A=DF1A/(12.D0*H)
      DF1B=DF1B/(6.D0*H)
      DIFF1=(4.D0*DF1B-DF1A)/3.D0
      DIFF2=DF2/(12.D0*H**2)
C     WRITE(5,'(1X,6(E13.6,1X))') DF1A,DF1B,DIFF1,DIFF2
C
      RETURN
      END   
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION GSRT(MH,MHSO4,MSO4,MTOT,ACTH,ACTHSO4,ACTSO4,OSM)
      IMPLICIT REAL*8(A-H,O-Z)
C
      PARAMETER(NCOEF=80)
C
      REAL*8 MH,MHSO4,MSO4,MTOT
C
      DUM=((ACTH**2*ACTSO4)*(MH**2*MSO4)/(4.D0*MTOT**3))
      ACTMEAN=DUM**(1.D0/3.D0)
      GSRT=3.D0*MTOT*(LOG(ACTMEAN)+1.D0) - (MH+MHSO4+MSO4)*OSM
C
C
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
c     SUBROUTINE PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
c     IMPLICIT REAL*8(A-H,O-Z)
C
c     PARAMETER(NCOEF=80,
c    >          TR=298.15D0)
C
c     REAL*8 COEFF(NCOEF),B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),
c    >       PSIC(5,5,5),PSIA(5,5,5)
C
c     D1=T-TR
c     D2=(T-TR)**2
C
C     2nd diff = a + bt 
C
c     B(1,1,1) = COEFF(1) + D1*COEFF(9)*1.D-3 
c    >          + 0.5D0*D2*(COEFF(29)*1.D-3 + COEFF(30)*D1*1.D-3/3.D0)
c     B(1,1,2) = COEFF(2) + D1*COEFF(10)*1.D-3 
c    >    + 0.5D0*D2*(COEFF(31)*1.D-3 + COEFF(32)*D1*1.D-3/3.D0)
c     B(1,1,3) = 0.D0
c     C(1,1)   = COEFF(3) + D1*COEFF(11)*1.D-3 
c    >    + 0.5D0*D2*(COEFF(33)*1.D-3 + COEFF(34)*D1*1.D-3/3.D0)
C
c     B(1,2,1) = COEFF(4) + D1*COEFF(12)*1.D-3 
c    >    + 0.5D0*D2*(COEFF(35)*1.D-3 + COEFF(36)*D1*1.D-3/3.D0)
c     B(1,2,2) = COEFF(5) + D1*COEFF(13)*1.D-3 
c    >    + 0.5D0*D2*(COEFF(37)*1.D-3 + COEFF(38)*D1*1.D-3/3.D0)
c     B(1,2,3) = 0.D0
c     C(1,2)   = COEFF(6) + D1*COEFF(14)*1.D-3 
c    >    + 0.5D0*D2*(COEFF(39)*1.D-3 + COEFF(40)*D1*1.D-3/3.D0)
C
c     C(2,1)   = COEFF(7) + D1*COEFF(41)*1.D-3 +
c    >                      0.5D0*D2*COEFF(43)*1.D-3
c     C(2,2)   = COEFF(8) + D1*COEFF(42)*1.D-3 +
c    >                      0.5D0*D2*COEFF(44)*1.D-3
C
c     THETAA(1,2)=0.D0
c     THETAA(2,1)=THETAA(1,2)
c     PSIA(1,2,1)=0.D0
c     PSIA(2,1,1)=PSIA(1,2,1)
C
C
c     RETURN
c     END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C 
      SUBROUTINE PARASET(T,COEFF,B,C,THETAC,THETAA,PSIC,PSIA)
      IMPLICIT REAL*8(A-H,O-Z)
C
      PARAMETER(NCOEF=80,
     >          TR=328.15D0)
C
      REAL*8 COEFF(NCOEF),B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),
     >       PSIC(5,5,5),PSIA(5,5,5)
C
      D1=T-TR
C
C....................................................................
C
      B(1,1,1)=COEFF(1) + D1*(COEFF(9)*1.D-3 + 
     >                    D1*(0.5D0*COEFF(29)*1.D-3 +
     >                        D1*(COEFF(30)*1.D-3/6.D0 + 
     >                            D1*(COEFF(73)*1.D-3/12.D0))))
      B(1,1,2)=COEFF(2) + D1*(COEFF(10)*1.D-3 + 
     >                    D1*(0.5D0*COEFF(31)*1.D-3 +
     >                        D1*(COEFF(32)*1.D-3/6.D0 + 
     >                            D1*(COEFF(74)*1.D-3/12.D0))))
      B(1,1,3)=0.D0
      C(1,1)=COEFF(3) + D1*(COEFF(11)*1.D-3 + 
     >                    D1*(0.5D0*COEFF(33)*1.D-3 +
     >                        D1*(COEFF(34)*1.D-3/6.D0 + 
     >                            D1*(COEFF(75)*1.D-3/12.D0))))
      C(2,1)=COEFF(7) + D1*(COEFF(41)*1.D-3 + 
     >                    D1*(0.5D0*COEFF(43)*1.D-3 +
     >                        D1*(COEFF(51)*1.D-3/6.D0 +
     >                            D1*(COEFF(76)*1.D-3/12.D0))))
C
      B(1,2,1)=COEFF(4) + D1*(COEFF(12)*1.D-3 + 
     >                    D1*(0.5D0*COEFF(35)*1.D-3 + 
     >                        D1*(COEFF(36)*1.D-3/6.D0 + 
     >                            D1*(COEFF(77)*1.D-3/12.D0))))
      B(1,2,2)=COEFF(5) + D1*(COEFF(13)*1.D-3 + 
     >                    D1*(0.5D0*COEFF(37)*1.D-3 +
     >                        D1*(COEFF(38)*1.D-3/6.D0 + 
     >                            D1*(COEFF(78)*1.D-3/12.D0))))
      B(1,2,3)=0.D0
      C(1,2)=COEFF(6) + D1*(COEFF(14)*1.D-3 + 
     >                    D1*(0.5D0*COEFF(39)*1.D-3 +
     >                        D1*(COEFF(40)*1.D-3/6.D0 + 
     >                            D1*(COEFF(79)*1.D-3/12.D0))))
      C(2,2)=COEFF(8) + D1*(COEFF(42)*1.D-3 + 
     >                    D1*(0.5D0*COEFF(44)*1.D-3 +
     >                        D1*(COEFF(52)*1.D-3/6.D0 +
     >                            D1*(COEFF(80)*1.D-3/12.D0))))
C
      THETAA(1,2)=0.D0
      THETAA(2,1)=THETAA(1,2)
      PSIA(1,2,1)=0.D0
      PSIA(2,1,1)=PSIA(1,2,1)
C
      RETURN
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION FFUNCS(CATM,ANM,A,B,C,THETAC,THETAA,PSIC,PSIA,XION,EDTHA,
     >                TEMP,CF)
       IMPLICIT REAL*8(A-H,O-Z)
       REAL*8 CATM(5),ANM(5)
       REAL*8 B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),PSIC(5,5,5),
     >        PSIA(5,5,5)
C
       DSQ=SQRT(XION)
       DUM=-A*(DSQ/(1.D0+1.2D0*DSQ)+(2.D0/1.2D0)*LOG(1.D0+1.2D0*DSQ))
       ALPHA1=2.D0
       ALPHAX=2.D0+(1.D0/TEMP-1.D0/298.15D0)*CF*100.D0
C
       SUMCA=CATM(1)*ANM(1)*B(1,1,2)*GDFUNC(ALPHA1,XION)/XION
     >      +CATM(1)*ANM(2)*B(1,2,2)*GDFUNC(ALPHAX,XION)/XION
C
       SUMCCAA=EDTHA*(CATM(1)*CATM(3)+CATM(2)*CATM(3)+ANM(1)*ANM(2))
C
       FFUNCS=DUM+SUMCA+SUMCCAA
      END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C                   
      SUBROUTINE ACT(A,B,C,THETAC,THETAA,PSIC,PSIA,CATM,ANM,NEUTM,
     >               XION,ACTH,ACTHSO4,ACTSO4,OSM,AWLG,TEMP,CF)
      IMPLICIT REAL*8(A-H,O-Z)
      REAL*8 CATM(5),ANM(5),NEUTM(5),MH,MHG,MPB,MHSO4,MSO4,
     >       B(5,5,3),C(5,5),THETAC(5,5),THETAA(5,5),PSIC(5,5,5),
     >       PSIA(5,5,5)
C
      ALPHA1=2.D0
      ALPHAX=2.D0+(1.D0/TEMP-1.D0/298.15D0)*CF * 100.D0
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
     >          TEMP,CF)
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
       END
C
C++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
