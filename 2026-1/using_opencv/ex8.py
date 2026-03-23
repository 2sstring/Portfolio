import cv2
import numpy as np
import matplotlib.pyplot as plt

# matplotlib이 새 창에 표시되도록 백엔드 설정
plt.switch_backend('TkAgg')

# Lenna.png 이미지 읽기
image = cv2.imread('Lenna.png')

# 이미지가 성공적으로 읽혔는지 확인
if image is None:
    print("Lenna.png 파일을 찾을 수 없습니다.")
else:
    # 원본 이미지 표시
    cv2.imshow('Original Image', image)
    
    # BGR 채널 분리 (OpenCV는 BGR 순서)
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    
    # 각 채널의 히스토그램 계산
    hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
    hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
    
    # matplotlib을 사용한 개별 채널 히스토그램 시각화 (개별 창에 표시)
    
    # Blue 채널 히스토그램
    plt.figure(figsize=(8, 6))
    plt.plot(hist_blue, color='blue')
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.savefig('blue_histogram_matplotlib.png')
    plt.show()
    
    # Green 채널 히스토그램
    plt.figure(figsize=(8, 6))
    plt.plot(hist_green, color='green')
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.savefig('green_histogram_matplotlib.png')
    plt.show()
    
    # Red 채널 히스토그램
    plt.figure(figsize=(8, 6))
    plt.plot(hist_red, color='red')
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.savefig('red_histogram_matplotlib.png')
    plt.show()
    
    # 각 채널 통계 정보 출력
    print(f"이미지 크기: {image.shape[1]}x{image.shape[0]}")
    print(f"총 픽셀 수: {image.size}")
    print("\n=== Blue Channel ===")
    print(f"평균: {np.mean(blue_channel):.2f}")
    print(f"표준편차: {np.std(blue_channel):.2f}")
    print(f"최소값: {np.min(blue_channel)}")
    print(f"최대값: {np.max(blue_channel)}")
    
    print("\n=== Green Channel ===")
    print(f"평균: {np.mean(green_channel):.2f}")
    print(f"표준편차: {np.std(green_channel):.2f}")
    print(f"최소값: {np.min(green_channel)}")
    print(f"최대값: {np.max(green_channel)}")
    
    print("\n=== Red Channel ===")
    print(f"평균: {np.mean(red_channel):.2f}")
    print(f"표준편차: {np.std(red_channel):.2f}")
    print(f"최소값: {np.min(red_channel)}")
    print(f"최대값: {np.max(red_channel)}")
    
    # 히스토그램 이미지 저장
    print("\n각 채널 히스토그램을 matplotlib으로 저장했습니다:")
    print("- blue_histogram_matplotlib.png")
    print("- green_histogram_matplotlib.png")
    print("- red_histogram_matplotlib.png")
