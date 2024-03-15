
C     PY ((2, 1), (1, 2, 22, 22)) : (2, 1, 2, 1, 22, 22) # M0_ 1
      SUBROUTINE SMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALE2,
     $  NHEL, ANS)
      IMPLICIT NONE
C     ALPHAS is given at scale2 (SHOULD be different of 0 for loop
C      induced, ignore for LO)  

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
      INTEGER PDGS(*)
      INTEGER NPDG, NHEL, PROCID
      DOUBLE PRECISION P(*)
      DOUBLE PRECISION ANS, ALPHAS, PI,SCALE2
      INCLUDE 'coupl.inc'


      IF (SCALE2.EQ.0)THEN
        PI = 3.141592653589793D0
        G = 2* DSQRT(ALPHAS*PI)
        CALL UPDATE_AS_PARAM()
      ELSE
        CALL UPDATE_AS_PARAM2(SCALE2, ALPHAS)
      ENDIF

      IF(2.EQ.PDGS(1).AND.1.EQ.PDGS(2).AND.2.EQ.PDGS(3)
     $ .AND.1.EQ.PDGS(4).AND.22.EQ.PDGS(5).AND.22.EQ.PDGS(6)
     $ .AND.(PROCID.LE.0.OR.PROCID.EQ.1)) THEN  ! 0
        CALL M0_SMATRIXHEL(P, NHEL, ANS)
      ENDIF

      RETURN
      END

      SUBROUTINE INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END


      SUBROUTINE CHANGE_PARA(NAME, VALUE)
      IMPLICIT NONE
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      CHARACTER*512 NAME
      DOUBLE PRECISION VALUE

      LOGICAL M0_HELRESET
      COMMON /M0_HELRESET/ M0_HELRESET

      INCLUDE '../Source/MODEL/input.inc'
      INCLUDE '../Source/MODEL/coupl.inc'

      M0_HELRESET = .TRUE.

      SELECT CASE (NAME)
      CASE ('CWWWL2')
      MDL_CWWWL2 = VALUE
      CASE ('DIM6_1')
      MDL_CWWWL2 = VALUE
      CASE ('CWL2')
      MDL_CWL2 = VALUE
      CASE ('DIM6_2')
      MDL_CWL2 = VALUE
      CASE ('CBL2')
      MDL_CBL2 = VALUE
      CASE ('DIM6_3')
      MDL_CBL2 = VALUE
      CASE ('CPWWWL2')
      MDL_CPWWWL2 = VALUE
      CASE ('DIM6_4')
      MDL_CPWWWL2 = VALUE
      CASE ('CPWL2')
      MDL_CPWL2 = VALUE
      CASE ('DIM6_5')
      MDL_CPWL2 = VALUE
      CASE ('aEWM1')
      AEWM1 = VALUE
      CASE ('SMINPUTS_1')
      AEWM1 = VALUE
      CASE ('Gf')
      MDL_GF = VALUE
      CASE ('SMINPUTS_2')
      MDL_GF = VALUE
      CASE ('aS')
      AS = VALUE
      CASE ('SMINPUTS_3')
      AS = VALUE
      CASE ('ymb')
      MDL_YMB = VALUE
      CASE ('YUKAWA_5')
      MDL_YMB = VALUE
      CASE ('ymt')
      MDL_YMT = VALUE
      CASE ('YUKAWA_6')
      MDL_YMT = VALUE
      CASE ('ymtau')
      MDL_YMTAU = VALUE
      CASE ('YUKAWA_15')
      MDL_YMTAU = VALUE
      CASE ('MZ')
      MDL_MZ = VALUE
      CASE ('MASS_23')
      MDL_MZ = VALUE
      CASE ('MT')
      MDL_MT = VALUE
      CASE ('MASS_6')
      MDL_MT = VALUE
      CASE ('MB')
      MDL_MB = VALUE
      CASE ('MASS_5')
      MDL_MB = VALUE
      CASE ('MH')
      MDL_MH = VALUE
      CASE ('MASS_25')
      MDL_MH = VALUE
      CASE ('MTA')
      MDL_MTA = VALUE
      CASE ('MASS_15')
      MDL_MTA = VALUE
      CASE ('MP')
      MDL_MP = VALUE
      CASE ('MASS_9000006')
      MDL_MP = VALUE
      CASE ('WZ')
      MDL_WZ = VALUE
      CASE ('DECAY_23')
      MDL_WZ = VALUE
      CASE ('WW')
      MDL_WW = VALUE
      CASE ('DECAY_24')
      MDL_WW = VALUE
      CASE ('WT')
      MDL_WT = VALUE
      CASE ('DECAY_6')
      MDL_WT = VALUE
      CASE ('WH')
      MDL_WH = VALUE
      CASE ('DECAY_25')
      MDL_WH = VALUE
      CASE ('WTau')
      MDL_WTAU = VALUE
      CASE ('DECAY_15')
      MDL_WTAU = VALUE
      CASE ('WH1')
      MDL_WH1 = VALUE
      CASE ('DECAY_9000006')
      MDL_WH1 = VALUE
      CASE DEFAULT
      WRITE(*,*) 'no parameter matching', NAME, VALUE
      END SELECT

      RETURN
      END

      SUBROUTINE UPDATE_ALL_COUP()
      IMPLICIT NONE
      CALL COUP()
      RETURN
      END


      SUBROUTINE GET_PDG_ORDER(PDG, ALLPROC)
      IMPLICIT NONE
CF2PY INTEGER, intent(out) :: PDG(1,6)
CF2PY INTEGER, intent(out) :: ALLPROC(1)
      INTEGER PDG(1,6), PDGS(1,6)
      INTEGER ALLPROC(1),PIDS(1)
      DATA PDGS/ 2,1,2,1,22,22 /
      DATA PIDS/ 1 /
      PDG = PDGS
      ALLPROC = PIDS
      RETURN
      END

      SUBROUTINE GET_PREFIX(PREFIX)
      IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: PREFIX(1)
      CHARACTER*20 PREFIX(1),PREF(1)
      DATA PREF / 'M0_'/
      PREFIX = PREF
      RETURN
      END



      SUBROUTINE SET_FIXED_EXTRA_SCALE(NEW_VALUE)
      IMPLICIT NONE
CF2PY logical, intent(in) :: new_value
      LOGICAL NEW_VALUE
      LOGICAL FIXED_EXTRA_SCALE
      INTEGER MAXJETFLAVOR
      DOUBLE PRECISION MUE_OVER_REF
      DOUBLE PRECISION MUE_REF_FIXED
      COMMON/MODEL_SETUP_RUNNING/MAXJETFLAVOR,FIXED_EXTRA_SCALE
     $ ,MUE_OVER_REF,MUE_REF_FIXED

      FIXED_EXTRA_SCALE = NEW_VALUE
      RETURN
      END

      SUBROUTINE SET_MUE_OVER_REF(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      LOGICAL FIXED_EXTRA_SCALE
      INTEGER MAXJETFLAVOR
      DOUBLE PRECISION MUE_OVER_REF
      DOUBLE PRECISION MUE_REF_FIXED
      COMMON/MODEL_SETUP_RUNNING/MAXJETFLAVOR,FIXED_EXTRA_SCALE
     $ ,MUE_OVER_REF,MUE_REF_FIXED

      MUE_OVER_REF = NEW_VALUE

      RETURN
      END

      SUBROUTINE SET_MUE_REF_FIXED(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      LOGICAL FIXED_EXTRA_SCALE
      INTEGER MAXJETFLAVOR
      DOUBLE PRECISION MUE_OVER_REF
      DOUBLE PRECISION MUE_REF_FIXED
      COMMON/MODEL_SETUP_RUNNING/MAXJETFLAVOR,FIXED_EXTRA_SCALE
     $ ,MUE_OVER_REF,MUE_REF_FIXED

      MUE_REF_FIXED = NEW_VALUE

      RETURN
      END


      SUBROUTINE SET_MAXJETFLAVOR(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      LOGICAL FIXED_EXTRA_SCALE
      INTEGER MAXJETFLAVOR
      DOUBLE PRECISION MUE_OVER_REF
      DOUBLE PRECISION MUE_REF_FIXED
      COMMON/MODEL_SETUP_RUNNING/MAXJETFLAVOR,FIXED_EXTRA_SCALE
     $ ,MUE_OVER_REF,MUE_REF_FIXED

      MAXJETFLAVOR = NEW_VALUE

      RETURN
      END


      SUBROUTINE SET_ASMZ(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      INTEGER NLOOP
      DOUBLE PRECISION ASMZ
      COMMON/A_BLOCK/ASMZ,NLOOP
      ASMZ = NEW_VALUE
      WRITE(*,*) 'asmz is set to ', NEW_VALUE

      RETURN
      END

      SUBROUTINE SET_NLOOP(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      INTEGER NLOOP
      DOUBLE PRECISION ASMZ
      COMMON/A_BLOCK/ASMZ,NLOOP
      NLOOP = NEW_VALUE
      WRITE(*,*) 'nloop is set to ', NEW_VALUE

      RETURN
      END


