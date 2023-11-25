"""
File used for my metric of Total Variation Norm + Distance from pure color
""" 
import jax
import jax.numpy as jnp

from functools import lru_cache, partial

@jax.jit
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


@jax.jit
def distFromPureColor2(image):
    """
    Just 0, 1 as my pure states
    """
    sqr0 = image**2
    sqr1 = (jnp.ones(image.shape) - image)**2
    min_ds = jnp.minimum(sqr0, sqr1)
    dist = jnp.sum(min_ds) / image.size
    return dist


@jax.jit
def scale01(x):
    return (x-jnp.min(x))/(jnp.max(x)-jnp.min(x))


@partial(jax.jit, static_argnames=['bot_perc', 'top_perc'])
def clip_img(image, bot_perc=20, top_perc=80):
    bot_val = jnp.percentile(image, bot_perc)
    top_val = jnp.percentile(image, top_perc)
    return jnp.clip(image, bot_val, top_val)


def closest_sq_dist_tv_mix(image, lambdaVal=0.5): 
    """
    A convex combination of total variation and square distance from the closest domain/pureColor
    sTV >> dPC most of the time in the trivial example tested
    """
    cliped_img = clip_img(image)
    scaledImg = scale01(cliped_img)
    sTV = scaledTotalVariation(scaledImg)
    dPC = distFromPureColor2(scaledImg)
    return lambdaVal*sTV + (1 - lambdaVal)*dPC