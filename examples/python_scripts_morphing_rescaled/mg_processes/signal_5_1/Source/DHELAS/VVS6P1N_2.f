C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     P(1,3)*P(2,1) + P(1,2)*P(2,3) - P(-1,1)*P(-1,3)*Metric(1,2) -
C      P(-1,2)*P(-1,3)*Metric(1,2)
C     
      SUBROUTINE VVS6P1N_2(V1, S3, COUP,V2)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      REAL*8 P1(0:3)
      REAL*8 P2(0:3)
      REAL*8 P3(0:3)
      COMPLEX*16 S3(*)
      COMPLEX*16 TMP2
      COMPLEX*16 TMP6
      COMPLEX*16 TMP8
      COMPLEX*16 TMP9
      COMPLEX*16 V1(*)
      COMPLEX*16 V2(6)
      P1(0) = DBLE(V1(1))
      P1(1) = DBLE(V1(2))
      P1(2) = DIMAG(V1(2))
      P1(3) = DIMAG(V1(1))
      P3(0) = DBLE(S3(1))
      P3(1) = DBLE(S3(2))
      P3(2) = DIMAG(S3(2))
      P3(3) = DIMAG(S3(1))
      V2(1) = +V1(1)+S3(1)
      V2(2) = +V1(2)+S3(2)
      P2(0) = -DBLE(V2(1))
      P2(1) = -DBLE(V2(2))
      P2(2) = -DIMAG(V2(2))
      P2(3) = -DIMAG(V2(1))
      TMP2 = (V1(3)*P2(0)-V1(4)*P2(1)-V1(5)*P2(2)-V1(6)*P2(3))
      TMP6 = (V1(3)*P3(0)-V1(4)*P3(1)-V1(5)*P3(2)-V1(6)*P3(3))
      TMP8 = (P1(0)*P3(0)-P1(1)*P3(1)-P1(2)*P3(2)-P1(3)*P3(3))
      TMP9 = (P2(0)*P3(0)-P2(1)*P3(1)-P2(2)*P3(2)-P2(3)*P3(3))
      V2(3)= COUP*S3(3)*(V1(3)*(+CI*(TMP8+TMP9))+(-CI*(P1(0)*TMP6+TMP2
     $ *P3(0))))
      V2(4)= COUP*S3(3)*(V1(4)*(-1D0)*(+CI*(TMP8+TMP9))+(+CI*(P1(1)
     $ *TMP6+TMP2*P3(1))))
      V2(5)= COUP*S3(3)*(V1(5)*(-1D0)*(+CI*(TMP8+TMP9))+(+CI*(P1(2)
     $ *TMP6+TMP2*P3(2))))
      V2(6)= COUP*S3(3)*(V1(6)*(-1D0)*(+CI*(TMP8+TMP9))+(+CI*(P1(3)
     $ *TMP6+TMP2*P3(3))))
      END


