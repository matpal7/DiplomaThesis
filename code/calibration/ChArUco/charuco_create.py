import cv2

# ------------------------------
# BOARD PARAMETERS
ARUCO_DICT = cv2.aruco.DICT_4X4_250

SQUARES_VERTICALLY = 8
SQUARES_HORIZONTALLY = 6
SQUARE_LENGTH = 0.035
MARKER_LENGTH = 0.024

# A3 size at 300 DPI
A3_WIDTH_PX = 3508
A3_HEIGHT_PX = 4961

MARGIN_PX = 100

SAVE_NAME = f"ChArUco_A3_{SQUARES_HORIZONTALLY}x{SQUARES_VERTICALLY}_sq{int(SQUARE_LENGTH*1000)}mm.png"# ------------------------------

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary
    )

    img = board.generateImage(
        (A3_WIDTH_PX, A3_HEIGHT_PX),
        marginSize=MARGIN_PX
    )

    cv2.imwrite(SAVE_NAME, img)
    cv2.imshow("ChArUco A3", img)
    cv2.waitKey(2000)

create_and_save_new_board()