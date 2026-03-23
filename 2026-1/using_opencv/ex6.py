import cv2
import numpy as np

# candies.png 이미지 읽기
image = cv2.imread('candies.png')

# 이미지가 성공적으로 읽혔는지 확인
if image is None:
    print("candies.png 파일을 찾을 수 없습니다.")
else:
    # 원본 이미지 표시
    cv2.imshow('Original Image', image)
    
    # BGR을 HSV로 변환 (빨간색 추출에 HSV가 더 효과적)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 빨간색의 HSV 범위 정의
    # 빨간색은 HSV에서 0과 179 근처에 두 개의 범위로 나뉨
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([179, 255, 255])
    
    # 빨간색 영역 마스크 생성
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    # 두 마스크 결합
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 마스크 표시
    cv2.imshow('Red Mask', red_mask)
    
    # 원본 이미지에서 빨간색만 추출
    red_candies = cv2.bitwise_and(image, image, mask=red_mask)
    
    # 빨간색 캔디만 표시
    cv2.imshow('Red Candies Only', red_candies)
    
    # 노이즈 제거를 위한形态學 처리
    kernel = np.ones((5,5), np.uint8)
    
    # 열림 연산으로 작은 노이즈 제거
    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # 닫힘 연산으로 빈 공간 채우기
    red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_CLOSE, kernel)
    
    # 정제된 마스크로 빨간색 추출
    red_candies_clean = cv2.bitwise_and(image, image, mask=red_mask_clean)
    
    cv2.imshow('Clean Red Candies', red_candies_clean)
    
    # 결과 이미지 저장
    cv2.imwrite('red_candies_result.png', red_candies_clean)
    print("빨간색 캔디 추출 결과를 red_candies_result.png로 저장했습니다.")
    
    # 키 입력 대기 (아무 키나 누르면 창이 닫힘)
    cv2.waitKey(0)
    
    # 모든 창 닫기
    cv2.destroyAllWindows()
