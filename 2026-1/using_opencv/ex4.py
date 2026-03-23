import cv2
import numpy as np

# Lenna.png 이미지 읽기
image = cv2.imread('Lenna.png')

# 이미지가 성공적으로 읽혔는지 확인
if image is None:
    print("Lenna.png 파일을 찾을 수 없습니다.")
else:
    # 이미지의 크기 정보 출력
    height, width, channels = image.shape
    print(f"이미지 크기: {width}x{height}, 채널 수: {channels}")
    
    # BGR 채널 분리 (OpenCV는 BGR 순서로 사용)
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    
    # 각 채널을 별도의 흑백 이미지로 표시
    cv2.imshow('Original Image', image)
    cv2.imshow('Blue Channel', blue_channel)
    cv2.imshow('Green Channel', green_channel)
    cv2.imshow('Red Channel', red_channel)
    
    # 각 채널을 컬러 이미지로 변환하여 표시
    # 파란색 채널만 남기기
    blue_only = np.zeros_like(image)
    blue_only[:, :, 0] = blue_channel
    cv2.imshow('Blue Only', blue_only)
    
    # 초록색 채널만 남기기
    green_only = np.zeros_like(image)
    green_only[:, :, 1] = green_channel
    cv2.imshow('Green Only', green_only)
    
    # 빨간색 채널만 남기기
    red_only = np.zeros_like(image)
    red_only[:, :, 2] = red_channel
    cv2.imshow('Red Only', red_only)
    
    # 키 입력 대기 (아무 키나 누르면 창이 닫힘)
    cv2.waitKey(0)
    
    # 모든 창 닫기
    cv2.destroyAllWindows()
