import cv2
import numpy as np

# Lenna.png 이미지 읽기
image = cv2.imread('Lenna.png')

# 이미지가 성공적으로 읽혔는지 확인
if image is None:
    print("Lenna.png 파일을 찾을 수 없습니다.")
else:
    # BGR을 YUV로 변환
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # 이미지의 크기 정보 출력
    height, width, channels = image.shape
    print(f"이미지 크기: {width}x{height}, 채널 수: {channels}")
    print("YUV 색상 모델:")
    print("Y (Luma): 휘도/밝기 (0-255)")
    print("U (Chroma Blue): 색차 정보 (0-255)")
    print("V (Chroma Red): 색차 정보 (0-255)")
    
    # YUV 채널 분리
    y_channel = yuv_image[:, :, 0]
    u_channel = yuv_image[:, :, 1]
    v_channel = yuv_image[:, :, 2]
    
    # 원본 이미지와 YUV 이미지 표시
    cv2.imshow('Original BGR Image', image)
    cv2.imshow('YUV Image', yuv_image)
    
    # 각 YUV 채널을 흑백 이미지로 표시
    cv2.imshow('Y Channel (Luma)', y_channel)
    cv2.imshow('U Channel (Chroma Blue)', u_channel)
    cv2.imshow('V Channel (Chroma Red)', v_channel)
    
    # 각 채널을 컬러로 시각화하기 위해 다시 BGR로 변환
    # Y 채널만 남기기 (U와 V는 128로 설정)
    y_only = np.zeros_like(yuv_image)
    y_only[:, :, 0] = y_channel
    y_only[:, :, 1] = 128  # 중간 색차 값
    y_only[:, :, 2] = 128  # 중간 색차 값
    y_color = cv2.cvtColor(y_only, cv2.COLOR_YUV2BGR)
    cv2.imshow('Y Only (Grayscale)', y_color)
    
    # U 채널만 남기기
    u_only = np.zeros_like(yuv_image)
    u_only[:, :, 1] = u_channel
    u_color = cv2.cvtColor(u_only, cv2.COLOR_YUV2BGR)
    cv2.imshow('U Only (Color)', u_color)
    
    # V 채널만 남기기
    v_only = np.zeros_like(yuv_image)
    v_only[:, :, 2] = v_channel
    v_color = cv2.cvtColor(v_only, cv2.COLOR_YUV2BGR)
    cv2.imshow('V Only (Color)', v_color)
    
    # 키 입력 대기 (아무 키나 누르면 창이 닫힘)
    cv2.waitKey(0)
    
    # 모든 창 닫기
    cv2.destroyAllWindows()
