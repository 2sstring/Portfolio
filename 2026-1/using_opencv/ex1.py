import cv2

# Lenna.png 이미지를 흑백으로 읽기
image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# 이미지가 성공적으로 읽혔는지 확인
if image is None:
    print("Lenna.png 파일을 찾을 수 없습니다.")
else:
    # 이미지 창에 표시
    cv2.imshow('Lenna Image', image)
    
    # 키 입력 대기 (아무 키나 누르면 창이 닫힘)
    cv2.waitKey(0)
    
    # 모든 창 닫기
    cv2.destroyAllWindows()
