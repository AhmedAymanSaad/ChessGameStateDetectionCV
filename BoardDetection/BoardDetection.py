"""
This class is responsible for detecting the board and segments it into squares.
"""
from Common.Common import *

#convert image to grayscale
def toGray(img):
    gray_img = rgb2gray(img)
    return gray_img

def autoThreshold(grey_image):
    # apply Otsu's automatic thresholding
    highthresh = threshold_otsu(grey_image)*0.5
    lowthresh = highthresh * 0.5
    
    return lowthresh,highthresh

#canny edge detection
def cannyEdge(img):
    LT ,HT = autoThreshold(img)
    edges = feature.canny(img ,sigma = math.sqrt(2), low_threshold =LT ,high_threshold=HT)
    #edges = feature.canny(img ,sigma = math.sqrt(2), low_threshold = 0.0188*5 ,high_threshold=0.0469*5 )
    return edges


#hough transform to detect lines
def houghTransform(img, lineOrientation):
    #H :hough transform accumulator (hough space)
    #theta: array of angles
    #d: array of distances
    #tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    if (lineOrientation == 0):
        tested_angles = np.linspace(0, np.pi / 2, 180, endpoint=False)
    else:
        tested_angles = np.linspace(-np.pi/2, 0, 180, endpoint=False)
    H, theta, d = transform.hough_line(img , theta=tested_angles)
    return H, theta, d

#Extract peaks from hough transform accumulator
def extractPeaks(H, theta, d):
    #parameters: 
    #threshold: Minimum intensity of peaks. Default is 0.5 * max(hspace).
    #min_distance (int) :Minimum distance separating lines (maximum filter size for first dimension of hough space)
    #min_angle (int) :Minimum angle separating lines (maximum filter size for second dimension of hough space).
    #num_peaks (int) :Maximum number of peaks. Default is the total number of peaks in hspace.
    threshold = math.ceil(0.005 * np.max(H))
    min_distance =math.ceil(H.shape[0]/70)
    min_angle = 0 #math.ceil(H.shape[1]/70)
    num_peaks = 30
    accums, angles, dists = transform.hough_line_peaks(H, theta, d, 
    threshold=threshold, min_distance=min_distance, min_angle=min_angle, num_peaks=num_peaks)
    return accums, angles, dists

#drawing the detected lines on the image
def drawLines(img, accums, angles, dists ,save = False):
    for angle, dist in zip(angles, dists):
        y0 = (dist) / np.sin(angle) 
        y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
        plt.plot((0, img.shape[1]), (y0, y1), '-r', linewidth=1)
    plt.imshow(img, cmap="gray")
    if save:
        plt.savefig('hlines.jpg')
    plt.show()

def drawHoughSpace(H, theta, d, Peak_accums, peak_angles, peak_dists):
    #draw rectangles on peaks in hough space
    for _, angle, dist in zip(Peak_accums, peak_angles, peak_dists):
        x = -1*np.rad2deg(angle)
        y = dist
        w = math.ceil(H.shape[1]/30)
        h = math.ceil(H.shape[0]/30)
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
    #draw hough space
    plt.imshow(np.log(H+1), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),d[-1], d[0]], cmap=cm.gray ,aspect='auto')
    plt.show()

def lineClustering(accums, angles, dists):
    
    X = np.column_stack((angles, dists))
    kmeans = KMeans(n_clusters=2, n_init= 20).fit(X)
    linesCluster1 = []
    linesCluster2 = []
    for i in range(len(kmeans.labels_)):
        line = [angles[i],dists[i]]
        if kmeans.labels_[i] == 0:
            linesCluster1.append(line)
        else:
            linesCluster2.append(line)
   
    return linesCluster1, linesCluster2

#draw clusters on hough
def drawClusters(H, theta, d, linesCluster1, linesCluster2):
    #draw rectangles on peaks in hough space
    for line in linesCluster1:
        angle = line[0]
        dist = line[1]
        x = -1*np.rad2deg(angle)
        y = dist
        w = math.ceil(H.shape[1]/30)
        h = math.ceil(H.shape[0]/30)
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
    for line in linesCluster2:
        angle = line[0]
        dist = line[1]
        x = -1*np.rad2deg(angle)
        y = dist
        w = math.ceil(H.shape[1]/30)
        h = math.ceil(H.shape[0]/30)
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect)
        
    #draw hough space
    plt.imshow(np.log(H+1), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),d[-1], d[0]], cmap=cm.gray ,aspect='auto')
    plt.show()

#get line equation from angle and distance
def getLineEquation(angle, dist):
    #y = mx + c
    m = -1*np.cos(angle)/np.sin(angle)
    c = dist/np.sin(angle)
    return m, c

def drawLinesonImage(img ,angles , dists):
    slopes = []
    y_intercepts = []
    for angle, dist in zip(angles, dists):
        m, c = getLineEquation(angle, dist)
        #print("m", m)
        slopes.append(m)
        y_intercepts.append(c)
        y0 = (dist) / np.sin(angle) 
        y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
        if(m > 0):
            plt.plot((0, img.shape[1]), (y0, y1), '-r', linewidth=1)
        else:
            plt.plot((0, img.shape[1]), (y0, y1), '-b', linewidth=1)
    meanSlope = np.median(slopes)
    sortedSlopes= sorted(slopes)
    print("sorted slopes", sortedSlopes)
    print("mean slope", meanSlope)
    plt.imshow(img)
    plt.show()
    return meanSlope ,slopes, y_intercepts
        
def plotpoints(m_array , c_array):
    plt.plot(m_array, c_array, 'ro') 
    plt.show()   


def draw_Bestfitline_OnHoughSpace(H ,theta ,d ,angles, dists):
    #draw best fit line on hough space
    x = -1*np.rad2deg(angles)
    y = dists
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    plt.imshow(np.log(H+1), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),d[-1], d[0]], cmap=cm.gray ,aspect='auto')
    plt.show()
    return z

def pointIsOnline(angle , dist ,line_eq):
    #line_eq: best fit line equation
    x = -1*np.rad2deg(angle)
    y = dist
    y1 = line_eq[0]*x + line_eq[1]
    if (y1 - y < 80):
        return True
    else:
        return False

# def outliersElimination(angles, dists):
#     #sort lines according to dists
#     lines = np.column_stack((angles, dists))
#     lines = lines[lines[:,1].argsort()]
#     #calculate the difference between each line and the next line
#     diff = np.diff(lines, axis=0)
#     #lines = lines[-15:]
#     print (lines)
#     print(diff)
#     return lines

# def outlierElimination(H ,theta, d, angles , dists):
#     bestFitline= draw_Bestfitline_OnHoughSpace(H ,theta ,d ,angles, dists)
#     newAngles = []
#     newDists = []
#     for angle, dist in zip(angles, dists):
#         if pointIsOnline(angle, dist, bestFitline):
#             newAngles.append(angle)
#             newDists.append(dist)
#     return newAngles, newDists

def outlierElimination(angles, dists, threshold_factor):
    # get mean angle for all lines 
    # if difference between line angle and mean angle is greater than threshold
    # then remove the line
    meanAngle = np.mean(angles)
    #threshold = avg difference between  line angle and mean angle
    threshold = threshold_factor * np.mean(np.abs(angles - meanAngle))
    # print("mean angle", meanAngle)
    # print("threshold", threshold)
    newAngles = []
    newDists = []
    for angle, dist in zip(angles, dists):
        diff = abs(angle - meanAngle)
        #print("diff", diff)
        if diff < threshold:
            newAngles.append(angle)
            newDists.append(dist)
    return newAngles, newDists 

# outlier lines elimination
def outlierElimination2(angles ,dists ,threshold_factor):
    # calcualte the avg distance to the nearest line for all lines
    # use this value as a threshold to eliminate outliers
    # if the distance of a line to its nearest line is greater than the threshold, then it is an outlier
    
    minDists = []
    for i in range(len(angles)):
        x1 = -1*np.rad2deg(angles[i])
        y1 = dists[i]
        minDist = 100000


        for j in range(len(angles)):
            if i != j:
                x2 = -1*np.rad2deg(angles[j])
                y2 = dists[j]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < minDist:
                    minDist = dist

                    
        minDists.append(minDist)
    threshold = np.mean(minDists) * threshold_factor
    print("threshold: " + str(threshold))

    newAngles = []
    newDists = []
    for i in range(len(angles)):
        print("angle ="+str(np.rad2deg(angles[i])), " dist= "+str(dists[i]))
        if minDists[i] < threshold:
            newAngles.append(angles[i])
            newDists.append(dists[i])
        
    return newAngles, newDists

#shift lines to be at 90 degrees
def shiftLines(angles, dists):
    diff_to_90 = -angles[0] + np.pi/2
    newAngles = []
    for angle in angles:
        newAngles.append(angle + diff_to_90)
    return newAngles, dists

def getLineEndpoints(img, angle ,dist):
    x1, y1, x2, y2 = 0, 0, 0, 0
    m,c =getLineEquation(angle, dist)
    if (c <= img.shape[1] and c >= 0):
        x1 = c
        y1 = 0
        x2 = img.shape[0]
        y2 = m*x2 + c
    else:
        y1 =0 
        x1 = (y1 - c)/m
        y2 = img.shape[1]
        x2 = (y2 - c)/m
    return x1, y1, x2, y2

def sortLines5(angles,dists,imgShape):
    horizontalLines = []
    verticalLines = []
    # get all lines intersecting with first line
    # if intersection point is in the image then it is a horizontal line
    # else it is a vertical line
    
    # get a line in the middle of the image
    num_lines = len(angles)
    fisrtlineIndex = int(num_lines/4)
    firstLine = [angles[fisrtlineIndex],dists[fisrtlineIndex]]
    m1,c1 = convertToYMXB(firstLine[0],firstLine[1])
    verticalLines.append(firstLine)
    # get all lines intersecting with first line
   
    for i in range(0,len(angles)):
        if i == fisrtlineIndex:
            continue
        m2,c2 = convertToYMXB(angles[i],dists[i])
        #calculate intersection point
        if m1 == math.inf:
            x = c1
            y = m2*x + c2
        elif m2 == math.inf:
            x = c2
            y = m1*x + c1
        else:
            x = (c2-c1)/(m1-m2)
            y = m1*x + c1
        # check if intersection point is in the image
        if x > imgShape[1]*0 and x < imgShape[1] and y > imgShape[0]*0 and y < imgShape[0]:
            horizontalLines.append([angles[i],dists[i]])
        else:
            verticalLines.append([angles[i],dists[i]])
    return horizontalLines, verticalLines

def convertToYMXB(angle, dist):
    # convert to y=mx+b
    # m = -(cos(theta)/sin(theta)))
    if math.sin(angle) == 0:
        return math.inf,math.inf
    m = -(math.cos(angle)/math.sin(angle))
    c = dist/math.sin(angle)
    return m,c

def removeRepeatedLines(angles,dists,imgShape):
    # calculate the angle between each line and its nearest line
    # if the angle is less than 10 degrees, then remove the line
    newAngles = []
    newDists = []
    linesNum = len(angles)
    clusters = []
    lines = []
    for i in range(len(angles)):
        lines.append([angles[i],dists[i]])

    for i in range(len(lines)):
        # remove line from lines
        if len(lines) == 0:
            break
        line = lines[0]
        lines.remove(line)
        cluster = []
        cluster.append(line)
        m1,c1 = convertToYMXB(line[0],line[1])
        minAngleAllowed = 0.0174533 * 7 # 1 degree
        for line2 in lines:
            m2,c2 = convertToYMXB(line2[0],line2[1])
            # calculate point of intersection of 2 lines
            if m1 == m2:
                continue
            x = (c2-c1)/(m1-m2)
            y = m1*x + c1
            # check if angle is inside the image
            if x < 0 or x > imgShape[1] or y < 0 or y > imgShape[0]:
                continue
            if 1+m1*m2 == 0:
                angle = math.pi/2
            else:
                angle = float(math.atan((m2-m1)/(1+m1*m2)))
            if abs(angle) < minAngleAllowed:
                #print("angle: " + str(angle) + " minAngleAllowed: " + str(minAngleAllowed))
                cluster.append(line2)
                lines.remove(line2)
        clusters.append(cluster)

    for cluster in clusters:
        newAngles.append(cluster[0][0])
        newDists.append(cluster[0][1])


    print("num of lines removed: " + str(len(angles) - len(newAngles)))
    return newAngles, newDists


# Sort lines
def sortLines(lines, originalImgSize):   

    # center: tuple of center row and center column
    center = (originalImgSize[0] / 2, originalImgSize[1] / 2)

    thetas = lines[0, :]
    rhos = lines[1, :]     

    linesCount = lines.shape[0]

    verLinesCount = np.sum(np.abs(thetas) < 45)

    if verLinesCount / linesCount > 0.5:
        dist = (-np.sin(np.deg2rad(thetas)) * center[0] + rhos) / np.cos(np.deg2rad(thetas)) - center[1] # vertical lines
    else:
        dist = (-np.cos(np.deg2rad(thetas)) * center[1] + rhos) / np.sin(np.deg2rad(thetas)) - center[0] # horizontal lines

    indexes_dists = sorted(list(enumerate(dist)), key = lambda val: val[1])
    indexes = [index for index, dist in indexes_dists]
    
    return lines[:, indexes]
    
def findHoughIntersections(horLines, verLines):
    horLinesCount = horLines.shape[1]
    verLinesCount = verLines.shape[1]

    # Contain all intersection points between the passed lines (horizontal, vertical)
    xIntersections = np.zeros((horLinesCount, verLinesCount))
    yIntersections = np.zeros((horLinesCount, verLinesCount))

    for i in range(horLinesCount):
        rho1 = horLines[1, i]
        theta1 = horLines[0, i]

        horLine = np.array([[np.cos(np.deg2rad(theta1))], [np.sin(np.deg2rad(theta1))], [-rho1]])

        for j in range(verLinesCount):
            rho2 = verLines[1, j]
            theta2 = verLines[0, j]

            verLine = np.array([[np.cos(np.deg2rad(theta2))], [np.sin(np.deg2rad(theta2))], [-rho2]])
            intersection = np.cross(horLine, verLine, axis=0) # get cross product of lines to get the intersection point
            intersection /= intersection[2, 0] # convert from 3D point to 2D point

            xIntersections[i, j] = intersection[0]
            yIntersections[i, j] = intersection[1]
    return (xIntersections, yIntersections)

def createReferenceIntersections(imgSize):
    xPoints = np.array([list(range(9))] * 9) * (imgSize / 8) + 1
    yPoints = np.transpose(xPoints)
    
    return (xPoints, yPoints)

def transfrom(transformation, xPoints, yPoints):
    x = transformation[0, 0] * xPoints + transformation[1, 0] * yPoints + transformation[2, 0]
    y = transformation[0, 1] * xPoints + transformation[1, 1] * yPoints + transformation[2, 1]
    z = transformation[0, 2] * xPoints + transformation[1, 2] * yPoints + transformation[2, 2]

    z[z == 0] = 0.00001 
    return (x / z, y / z)

def geoTransformation(xRef, yRef, xIntersections, yIntersections):
    cornersRef = np.array([
        [xRef[0, 0], yRef[0, 0]],
        [xRef[0, -1], yRef[0, -1]],
        [xRef[-1, -1], yRef[-1, -1]],
        [xRef[-1, 0], yRef[-1, 0]]
    ])


    minDist = 10 
    bestMatchesCount = 0 
    bestAverageError = 1e10
    bestCorners = None

    
    horLinesCount = xIntersections.shape[0] 
    verLinesCount = xIntersections.shape[1] 

    roundedHor = round(horLinesCount * 0.6)
    roundedVer = round(verLinesCount * 0.6)

    for i1 in range(min(roundedHor, horLinesCount)):
        for i2 in range(horLinesCount - 1, max(horLinesCount - roundedHor, i1 + 8) - 1, -1):
            for j1 in range(min(roundedVer, verLinesCount)):
                for j2 in range(verLinesCount - 1, max(verLinesCount - roundedVer, j1 + 8) - 1, -1):

                    if i1 == i2 or j1 == j2: continue
                    

                    corners = np.array([
                        [xIntersections[i1, j1], yIntersections[i1, j1]],
                        [xIntersections[i1, j2], yIntersections[i1, j2]],
                        [xIntersections[i2, j2], yIntersections[i2, j2]],
                        [xIntersections[i2, j1], yIntersections[i2, j1]]
                    ])

                    dist1 = corners[1] - corners[0]
                    dist2 = corners[2] - corners[0]
                    if dist1[0] * dist2[1] - dist1[1] * dist2[0] < 0:
                        corners[1], corners[3] = corners[3].copy(), corners[1].copy()

                    cornersMag = corners[:, 0] ** 2 + corners[:, 1] ** 2
                    index = [index for index, corner in enumerate(cornersMag) if corner == min(cornersMag)][0]
                    corners = np.roll(corners, -index, axis = 0)

                    transformation = transform.ProjectiveTransform()
                    transformation.estimate(corners, cornersRef)
                    
                    xIntersectionsFlattened = xIntersections.flatten(order = 'F')
                    yIntersectionsFlattened = yIntersections.flatten(order = 'F')
                    xRefFlattened = xRef.flatten(order = 'F')
                    yRefFlattened = yRef.flatten(order = 'F')

                    xIntersectionsFlattened.resize((len(xIntersectionsFlattened), 1))
                    yIntersectionsFlattened.resize((len(yIntersectionsFlattened), 1))

                    xRefFlattened.resize((len(xRefFlattened), 1))
                    yRefFlattened.resize((len(yRefFlattened), 1))

                    intersections = transfrom(transformation.params.transpose(), xIntersectionsFlattened, yIntersectionsFlattened)
                    intersectionsRef = (xRefFlattened, yRefFlattened)
                    
                    intersections = np.array(list(zip(intersections[0].flatten(), intersections[1].flatten())))
                    intersectionsRef = np.array(list(zip(intersectionsRef[0].flatten(), intersectionsRef[1].flatten())))

                    points = 1e6 * np.ones((intersectionsRef.shape[0], 1))
                    for i in range(intersectionsRef.shape[0]):
                        x = intersectionsRef[i, 0]
                        y = intersectionsRef[i, 1]
                        d = np.sqrt((x - intersections[:, 0]) ** 2 + (y - intersections[:, 1]) ** 2)
                        points[i] = np.min(d)

                    matchesCount = np.sum(points < minDist)
                    averageError = np.mean(points[points < minDist])
                    
                    if matchesCount < bestMatchesCount: continue
                    if matchesCount == bestMatchesCount and averageError > bestAverageError: continue
                    
                    bestAverageError = averageError
                    bestMatchesCount = matchesCount
                    bestCorners = corners
                    
    return bestCorners, bestAverageError, bestMatchesCount

def calcIntersections(corners):
    dist1 = corners[1] - corners[0]
    dist2 = corners[2] - corners[0]

    if dist1[0] * dist2[1] - dist1[1] * dist2[0] < 0:
        corners[1], corners[3] = corners[3].copy(), corners[1].copy()


    correctedCorners = corners
    correctedCorners = correctedCorners[:, 0] ** 2 + correctedCorners[:, 1] ** 2
    indecies = [index for index, corner in enumerate(correctedCorners) if corner == min(correctedCorners)]
    index = indecies[0]
    correctedCorners = np.roll(corners, -index, axis=0)
    
    xIntersectionsRef = np.array([list(range(9))] * 9) * 1 + 1
    yIntersectionsRef = np.transpose(xIntersectionsRef)

    cornersRef = np.array([
        [1, 1],
        [9, 1],
        [9, 9],
        [1, 9]
    ])


    transformation = transform.ProjectiveTransform()
    transformation.estimate(cornersRef, correctedCorners)

    xIntersectionsRefFlattened = xIntersectionsRef.flatten(order='F')
    yIntersectionsRefFlattened = yIntersectionsRef.flatten(order='F')
   
    intersections = np.array(list(zip(xIntersectionsRefFlattened, yIntersectionsRefFlattened, np.ones((len(yIntersectionsRefFlattened), 1)))), dtype = 'object')

    transformed = np.matmul(transformation.params, intersections.transpose())
    intersections = np.zeros((2, len(yIntersectionsRefFlattened)))
    intersections[0, :] = transformed[0, :] / transformed[2, :]
    intersections[1, :] = transformed[1, :] / transformed[2, :]

    return intersections

def writeCorners(corners, filePath):
    with open(filePath, 'w') as f:
        for point in corners:
            f.write(f'{point[0]},{point[1]}\n')

def writeIntersections(intersections, filePath): 
    with open(filePath, 'w') as f:
        for axis in intersections:
            for i in range(len(axis)):
                f.write(f'{axis[i]}' + ('' if i == len(axis) - 1 else ','))
            f.write('\n')

def detectBoard(img, plot=False):
    """
    This function is responsible for detecting the board.
    :return: The detected board.
    """
    imgSize = img.shape

    processing_img = img
    copy_img = processing_img.copy()
    gray_img = toGray(processing_img)

    edges = cannyEdge(gray_img)

    H, theta, d = transform.hough_line(edges)
    H1, theta1, d1 = houghTransform(edges,1) #1 for vertical lines
    H2, theta2, d2 = houghTransform(edges,0) #0 for horizontal lines

    accums1, angles1, dists1 = extractPeaks(H1, theta1, d1)
    accums2,angles2, dists2 = extractPeaks(H2,theta2,d2)

    #combine accums , angles and dists
    accums = np.concatenate((accums1,accums2))
    angles = np.concatenate((angles1,angles2))
    dists = np.concatenate((dists1,dists2))

    #drawLines(copy_img, accums, angles, dists)
    if plot:
        drawHoughSpace(H,theta,d,accums, angles, dists)

    # REMOVE REPEATED LINES
    angles, dists = removeRepeatedLines(angles, dists,processing_img.shape)
    temp_angles = []
    while len(angles) != len(temp_angles):
        temp_angles = angles
        angles, dists = removeRepeatedLines(angles, dists,processing_img.shape)
    if plot:
        drawLines(copy_img, accums, angles, dists)
        drawHoughSpace(H,theta,d,accums, angles, dists)

    # SORT LINES INTO HORIZONTAL AND VERTICAL
         
    sortedHorizontalLines, sortedVerticalLines = sortLines5(angles, dists,processing_img.shape)
    if plot:
        drawClusters(H,theta,d,sortedHorizontalLines,sortedVerticalLines)
    angles_H = np.array(sortedHorizontalLines)[:,0]
    dists_H = np.array(sortedHorizontalLines)[:,1]
    #drawLines(copy_img, accums, angles_H, dists_H)
    angles_V = np.array(sortedVerticalLines)[:,0]
    dists_V = np.array(sortedVerticalLines)[:,1]
    #drawLines(copy_img, accums, angles_V, dists_V)

    # REMOVE OUTLIERS
    angles_H, dists_H = outlierElimination(angles_H,dists_H,2.1)
    if plot:
        drawLines(copy_img, accums, angles_H, dists_H)
        
    angles_V, dists_V = outlierElimination(angles_V, dists_V,2.1)
    if plot:
        drawLines(copy_img, accums, angles_V, dists_V)

    ########
    
    lines_H = np.array([np.rad2deg(angles_H), dists_H])
    lines_V = np.array([np.rad2deg(angles_V), dists_V])

    referenceImgSize = 319

    lines_H = sortLines(lines_H, imgSize)
    lines_V = sortLines(lines_V, imgSize)

    # for board2
    
    # angles_H = np.deg2rad(lines_H[0, :-1])
    # dists_H = lines_H[1, :-1]
    # angles_V = np.deg2rad(lines_V[0, 1:-1])
    # dists_V = lines_V[1, 1:-1]

    # lines_H = np.array([lines_H[0, :-1], lines_H[1, :-1]])
    # lines_V = np.array([lines_V[0, 1:-1], lines_V[1, 1:-1]])

    # drawLines(copy_img, accums, angles_H, dists_H)
    # drawLines(copy_img, accums, angles_V, dists_V)

    xIntersections, yIntersections = findHoughIntersections(lines_H, lines_V)
    xIntersectionsRef, yIntersectionsRef = createReferenceIntersections(referenceImgSize)
    corners, err, matches = geoTransformation(xIntersectionsRef, yIntersectionsRef, xIntersections, yIntersections)
    intersections = calcIntersections(corners)

    # writeCorners(corners, f'{directory}/corners.csv')
    # writeIntersections(intersections, f'{directory}/intersections.csv')
    
    return corners, intersections


