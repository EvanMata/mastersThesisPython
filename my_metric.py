"""
File used for my metric of Total Variation Norm + Distance from pure color
""" 

import jax.numpy as jnp

def scaledTotalVariation(scaledImg): 
    # Calculate the scaled total variation ( scaled to 0-1 )
    diffX = jnp.abs(jnp.diff(scaledImg, axis=1))
    diffY = jnp.abs(jnp.diff(scaledImg, axis=0))
    scaledTV = (jnp.sum(diffX) / diffX.size) + (jnp.sum(diffY) / diffY.size)
    return scaledTV


def distFromPureColor(image, pureColors=[0, 1]):
    # There's got to be a more efficient way of doing this.
    """
    For each pixel, find its closest class (ie pureColor, ie magnetic=1, nonMagentic=0)
    then take the distance from that pure color squared. Finally, normalize by
    the worst possible value to be [0,1] range
    """
    closestColors = jnp.zeros(image.shape)
    pureColors = sorted(pureColors)
    colorMidpoints = [jnp.mean(jnp.array([pureColors[i], pureColors[i+1]])) for i in
                      range(len(pureColors) - 1)]
    previouslyUnseen = jnp.ones(image.shape)
    justSeen = jnp.ones(image.shape)

    for i in range(len(colorMidpoints)):
        colorMidpoint = colorMidpoints[i]
        previouslyUnseen = jnp.where(image < colorMidpoint, 0, 1)
        newInfo = justSeen - previouslyUnseen #Just seen is 1's
        currentColorArr = jnp.ones(image.shape) * pureColors[i]
        closestColors = jnp.where(newInfo >= 1, currentColorArr, closestColors)

        justSeen = jnp.copy(previouslyUnseen)
        if i == len(colorMidpoints) - 1:
            closestColors = jnp.where(image >= colorMidpoint, pureColors[i+1], closestColors)
            previouslyUnseen = jnp.where(image < colorMidpoint, 0, 1)
            # ^Not necessary, aside from sanity check

    distFromClosests = jnp.abs(image - closestColors)**2
    distOverallReduced = jnp.sum(distFromClosests) / image.size

    biggestPossibleGapInfo = [pureColors[0]] + colorMidpoints + [pureColors[-1]]
    allGaps = jnp.array([biggestPossibleGapInfo[i+1] - biggestPossibleGapInfo[i] for i in
                      range(len(biggestPossibleGapInfo) - 1)])
    largestPossiGap = jnp.max(allGaps)
    normFactor = largestPossiGap**2
    distOverall = distOverallReduced / normFactor

    return distOverall

def closest_sq_dist_tv_mix(image, lambdaVal=0.5): 
    """
    A convex combination of total variation and square distance from the closest domain/pureColor
    sTV >> dPC most of the time in the trivial example tested
    """
    scaledImg = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image))
    sTV = scaledTotalVariation(scaledImg)
    dPC = distFromPureColor(scaledImg, pureColors=[0, 1])
    return lambdaVal*sTV + (1 - lambdaVal)*dPC