from cv2 import VideoCapture, cvtColor, Canny, COLOR_BGR2GRAY, HoughLinesP
from cv2 import threshold, THRESH_BINARY, dilate, floodFill, HoughLines, erode, VideoWriter
from numpy import pi, ones, zeros, uint8, where, cos, sin, sqrt

def determinant(a, b):
    return a[0] * b[1] - a[1] * b[0]

def findIntersection(line1, line2, xStart, yStart, xEnd, yEnd):
    xDiff = (line1[0][0]-line1[1][0],line2[0][0]-line2[1][0])
    yDiff = (line1[0][1]-line1[1][1],line2[0][1]-line2[1][1])
    div = determinant(xDiff, yDiff)
    if div == 0:
        return None
    d = (determinant(*line1), determinant(*line2))
    x = int(determinant(d, xDiff) / div)
    y = int(determinant(d, yDiff) / div)
    if (x<xStart) or (x>xEnd):
        return None
    if (y<yStart) or (y>yEnd):
        return None
    return x,y

def court_detection_points(frame,width,height):
    xOLeftLine = None
    xORightLine = None
    yOTopLine = None
    yOBottomLine = None
    xFLeftLine = None
    xFRightLine = None
    yFTopLine = None
    yFBottomLine = None

    extraLen = width/3
    axis_top = [[-extraLen,0],[width+extraLen,0]]
    axis_right = [[width+extraLen,0],[width+extraLen,height]]
    axis_bottom = [[-extraLen,height],[width+extraLen,height]]
    axis_left = [[-extraLen,0],[-extraLen,height]]

    gry = cvtColor(frame, COLOR_BGR2GRAY)
    bw = threshold(gry, 156, 255, THRESH_BINARY)[1]
    canny = Canny(bw, 100, 200)
        
        # Using hough lines probablistic to find lines with most intersections
    hPLines = HoughLinesP(canny, 1, pi/180, threshold=150, minLineLength=100, maxLineGap=10)
    if type(hPLines) == type(None):
        return None
    intersectNum = zeros((len(hPLines),2))
    i = 0
    for hPLine1 in hPLines:
        Line1x1, Line1y1, Line1x2, Line1y2 = hPLine1[0]
        Line1 = [[Line1x1,Line1y1],[Line1x2,Line1y2]]
        for hPLine2 in hPLines:
            Line2x1, Line2y1, Line2x2, Line2y2 = hPLine2[0]
            Line2 = [[Line2x1,Line2y1],[Line2x2,Line2y2]]
            if Line1 is Line2:
                continue
            if Line1x1>Line1x2:
                temp = Line1x1
                Line1x1 = Line1x2
                Line1x2 = temp
                
            if Line1y1>Line1y2:
                temp = Line1y1
                Line1y1 = Line1y2
                Line1y2 = temp
                    
            intersect = findIntersection(Line1, Line2, Line1x1-200, Line1y1-200, Line1x2+200, Line1y2+200)
            if intersect is not None:
                intersectNum[i][0] += 1
        intersectNum[i][1] = i
        i += 1

        # Lines with most intersections get a fill mask command on them
    i = p = 0
    dilation = dilate(bw, ones((5, 5), uint8), iterations=1)
    nonRectArea = dilation.copy()
    intersectNum = intersectNum[(-intersectNum)[:, 0].argsort()]
    if len(intersectNum)<8:
        return None
    for hPLine in hPLines:
        x1,y1,x2,y2 = hPLine[0]
            # line(frame, (x1,y1), (x2,y2), (255, 255, 0), 2)
        for p in range(8):
            if (i==intersectNum[p][1]) and (intersectNum[i][0]>0):
                    # line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
                floodFill(nonRectArea, zeros((height+2, width+2), uint8), (x1, y1), 1) 
                floodFill(nonRectArea, zeros((height+2, width+2), uint8), (x2, y2), 1) 
        i+=1
    dilation[where(nonRectArea == 255)] = 0
    dilation[where(nonRectArea == 1)] = 255
    eroded = erode(dilation, ones((5, 5), uint8)) 
    cannyMain = Canny(eroded, 90, 100)
        
        # Extreme lines found every frame
    xOLeft = width + extraLen
    xORight = 0 - extraLen
    xFLeft = width + extraLen
    xFRight = 0 - extraLen
        
    yOTop = height
    yOBottom = 0
    yFTop = height
    yFBottom = 0
        
        # Finding all lines then allocate them to specified extreme variables
    hLines = HoughLines(cannyMain, 2, pi/180, 300)
    for hLine in hLines:
        for rho,theta in hLine:
            a = cos(theta)
            b = sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + width*(-b))
            y1 = int(y0 + width*(a))
            x2 = int(x0 - width*(-b))
            y2 = int(y0 - width*(a))
                
                # Furthest intersecting point at every axis calculations done here
            intersectxF = findIntersection(axis_bottom, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectyO = findIntersection(axis_left, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectxO = findIntersection(axis_top, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectyF = findIntersection(axis_right, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
                
            if (intersectxO is None) and (intersectxF is None) and (intersectyO is None) and (intersectyF is None):
                continue
                
            if intersectxO is not None:
                if intersectxO[0] < xOLeft:
                    xOLeft = intersectxO[0]
                    xOLeftLine = [[x1,y1],[x2,y2]]
                if intersectxO[0] > xORight:
                    xORight = intersectxO[0]
                    xORightLine = [[x1,y1],[x2,y2]]
            if intersectyO is not None:
                if intersectyO[1] < yOTop:
                    yOTop = intersectyO[1]
                    yOTopLine = [[x1,y1],[x2,y2]]
                if intersectyO[1] > yOBottom:
                    yOBottom = intersectyO[1]
                    yOBottomLine = [[x1,y1],[x2,y2]]
                        
            if intersectxF is not None:
                if intersectxF[0] < xFLeft:
                    xFLeft = intersectxF[0]
                    xFLeftLine = [[x1,y1],[x2,y2]]
                if intersectxF[0] > xFRight:
                    xFRight = intersectxF[0]
                    xFRightLine = [[x1,y1],[x2,y2]]
            if intersectyF is not None:
                if intersectyF[1] < yFTop:
                    yFTop = intersectyF[1]
                    yFTopLine = [[x1,y1],[x2,y2]]
                if intersectyF[1] > yFBottom:
                    yFBottom = intersectyF[1]
                    yFBottomLine = [[x1,y1],[x2,y2]]
                # line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
        
        # lineEndpoints = []
        # lineEndpoints.append(xOLeftLine)
        # lineEndpoints.append(xORightLine)
        # lineEndpoints.append(yOTopLine)
        # lineEndpoints.append(yOBottomLine)
        # lineEndpoints.append(xFLeftLine)
        # lineEndpoints.append(xFRightLine)
        # lineEndpoints.append(yFTopLine)
        # lineEndpoints.append(yFBottomLine)
        
        # for i in range(len(lineEndpoints)):
        #     line(frame, (lineEndpoints[i][0][0],lineEndpoints[i][0][1]), (lineEndpoints[i][1][0],lineEndpoints[i][1][1]), (0, 0, 255), 2)
    
    if xOLeftLine is None or xORightLine is None or yOTopLine is None or yOBottomLine is None or xFLeftLine is None or xFRightLine is None or yFTopLine is None or yFBottomLine is None:
        return None

        # Top line has margin of error that effects all courtmapped outputs 
    yOTopLine[0][1] = yOTopLine[0][1]+4
    yOTopLine[1][1] = yOTopLine[1][1]+4
        
    yFTopLine[0][1] = yFTopLine[0][1]+4
    yFTopLine[1][1] = yFTopLine[1][1]+4
        
        # Find four corners of the court and display it
    topLeftP = findIntersection(xOLeftLine, yOTopLine, -extraLen, 0, width+extraLen, height)
    topRightP = findIntersection(xORightLine, yFTopLine, -extraLen, 0, width+extraLen, height)
    bottomLeftP = findIntersection(xFLeftLine, yOBottomLine, -extraLen, 0, width+extraLen, height)
    bottomRightP = findIntersection(xFRightLine, yFBottomLine, -extraLen, 0, width+extraLen, height)
    
    return [topLeftP,topRightP,bottomLeftP,bottomRightP]