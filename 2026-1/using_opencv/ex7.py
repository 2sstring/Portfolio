import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lenna.png 이미지 읽기
image = cv2.imread('Lenna.png')

# 이미지가 성공적으로 읽혔는지 확인
if image is None:
    print("Lenna.png 파일을 찾을 수 없습니다.")
else:
    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 원본 이미지와 그레이스케일 이미지 표시
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    
    # 히스토그램 계산
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # 히스토그램 그래프 그리기 (matplotlib 사용)
    plt.figure(figsize=(10, 6))
    
    # 히스토그램 플롯
    plt.subplot(1, 2, 1)
    plt.plot(hist, color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    
    # 히스토그램 막대 그래프
    plt.subplot(1, 2, 2)
    plt.bar(range(256), hist.flatten(), color='black', width=1.0)
    plt.title('Grayscale Histogram (Bar)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # OpenCV를 사용한 히스토그램 시각화
    hist_image = np.zeros((300, 512, 3), dtype=np.uint8)
    
    # 히스토그램 정규화
    cv2.normalize(hist, hist, 0, hist_image.shape[0], cv2.NORM_MINMAX)
    
    # 히스토그램 그리기
    for i in range(1, 256):
        cv2.line(hist_image, (i-1, hist_image.shape[0] - int(hist[i-1])), 
                (i, hist_image.shape[0] - int(hist[i])), (255, 255, 255), 2)
    
    cv2.imshow('Histogram (OpenCV)', hist_image)
    
    # 히스토그램 통계 정보 출력
    print(f"이미지 크기: {gray_image.shape[1]}x{gray_image.shape[0]}")
    print(f"총 픽셀 수: {gray_image.size}")
    print(f"평균 밝기: {np.mean(gray_image):.2f}")
    print(f"표준편차: {np.std(gray_image):.2f}")
    print(f"최소 밝기: {np.min(gray_image)}")
    print(f"최대 밝기: {np.max(gray_image)}")
    
    # 히스토그램 이미지 저장
    cv2.imwrite('histogram_opencv.png', hist_image)
    print("히스토그램을 histogram_opencv.png로 저장했습니다.")
    
    # 키 입력 대기 (아무 키나 누르면 창이 닫힘)
    cv2.waitKey(0)
    
    # 모든 창 닫기
    cv2.destroyAllWindows()
