funLogLikelihood <- function(parVec, mutationMat, q = NULL){
	gamma <- parVec[1]
	delta <- parVec[-1]
	M <- ncol(mutationMat)
	if(is.null(q))q <- as.vector(table(factor(rowSums(mutationMat), level = 0:M)))
	if(length(delta) == 1){
		J <- 1:M
		logL <- M * q[1] * log(1 - delta) + q[1] * log(1 - gamma) + sum(q[-1] * (log(J / M) + (J - 1) * log(delta) + (M - J) * log(1 - delta) + log(M / J * delta * (1 - gamma) + gamma)))
	}else if(length(delta) == M){
		tempA <- matrix(NA, 2^M, M)
		for(i in 1:M)tempA[, i] <- rep(rep(c(FALSE, TRUE), each = 2^(M - i)), 2^(i - 1))
		tempProbMat <- matrix(rep(delta, each = 2^M), 2^M, M)
		tempProbMat[!tempA] <- 1 - tempProbMat[!tempA]
		tempProd <- apply(tempProbMat, 1, prod)
		tempMat <- rep(delta / sum(delta), each = 2^M) * tempA * (tempProd / tempProbMat)
		tempProb <- (1 - gamma) * tempProd + gamma * rowSums(tempMat)
		prob <- tempProb[colSums(t(mutationMat) * 2^((M - 1):0)) + 1]
		logL <- sum(log(prob))
	}else stop("Invalid Value of delta")
	return(-logL)
}

funGradient <- function(parVec, mutationMat, q = NULL){
	gamma <- parVec[1]
	delta <- parVec[-1]
	M <- ncol(mutationMat)
	if(is.null(q))q <- as.vector(table(factor(rowSums(mutationMat), level = 0:M)))
	if(length(delta) == 1)delta <- rep(delta, M)
	if(length(delta) != M)stop("Invalid Value of delta")
	sumDelta <- sum(delta)
	tempA <- matrix(NA, 2^M, M)
	for(i in 1:M)tempA[, i] <- rep(rep(c(FALSE, TRUE), each = 2^(M - i)), 2^(i - 1))
	tempProbMat1 <- tempProbMat2 <- matrix(delta, 2^M, M, byrow = TRUE)
	tempProbMat2[!tempA] <- 1 - tempProbMat1[!tempA]
	temp1 <- apply(tempProbMat2, 1, prod)
	temp2 <- temp1 / tempProbMat2
	tempProbMat3 <- tempProbMat1 / sumDelta * tempA * temp2
	temp3 <- rowSums(tempProbMat3)
	tempProb <- (1 - gamma) * temp1 + gamma * temp3
	tempGradGamma <- (temp3 - temp1) / tempProb
	tempGradMat <- matrix(NA, 2^M, M)
	temp4 <- (1 - gamma) * (tempA * 2 - 1) * temp2
	temp5 <- (sumDelta - delta) / sumDelta^2
	temp6 <- rep(temp5, each = 2^M) * tempA * temp2
	for(i in 1:M){
		temp7 <- -tempProbMat1[, -i, drop = FALSE] / sumDelta^2 * tempA[, -i, drop = FALSE] * temp2[, -i, drop = FALSE] + tempProbMat1[, -i, drop = FALSE] / sumDelta * tempA[, -i, drop = FALSE] * (temp2[, -i, drop = FALSE] / tempProbMat2[, i] * (tempA[, i] * 2 - 1))
		tempGradMat[, i] <- temp4[, i] + gamma * (temp6[, i] + rowSums(temp7))
	}
	tempGradMat <- tempGradMat / tempProb
	tempIdx <- colSums(t(mutationMat) * 2^((M - 1):0)) + 1
	grad <- c(sum(tempGradGamma[tempIdx]), colSums(tempGradMat[tempIdx, ]))
	return(-grad)
}

funEstimate <- function(mutationMat, tol = 1e-7){
	N <- nrow(mutationMat)
	M <- ncol(mutationMat)
	tempRowSums <- rowSums(mutationMat)
	q <- as.vector(table(factor(tempRowSums, level = 0:M)))
	piHat <- colMeans(mutationMat)
	probMat <- matrix(rep(piHat, each = N), N, M)
	tempMat <- log(probMat^mutationMat * (1 - probMat)^(!mutationMat))
	logL0Each <- rowSums(tempMat)
	logL0 <- sum(logL0Each)
	for(eps in 10^(-((-log10(tol)):3))){
		tempOptim <- try(optim(c(mean(tempRowSums > 0), piHat), funLogLikelihood, gr = funGradient, method = "L-BFGS-B", lower = c(0, rep(eps, M)), upper = rep(1 - eps, M + 1), mutationMat = mutationMat, q = q), silent = TRUE)
		if(class(tempOptim) == "list" && tempOptim$convergence == 0 && all(tempOptim$par > 0) && all(tempOptim$par < 1))break
	}
	if(class(tempOptim) == "try-error")stop("ERROR in OPTIM FUNCTION!")
	logL1 <- -tempOptim$value
	gammaHat <- tempOptim$par[1]
	deltaHat <- tempOptim$par[-1]
	if(logL1 < logL0){gammaHat <- 0; deltaHat <- piHat; logL1 <- logL0}
	S <- -2 * (logL0 - logL1)
	return(list(pi = piHat, gamma = gammaHat, delta = deltaHat, logL0 = logL0, logL1 = logL1, S = S))
}

funGlobalTest <- function(mutationMat, maxSSimu = NULL, nSimu = 1000, nPairStart = 10, maxSize = 6, detail = TRUE){
      N <- nrow(mutationMat)
	M <- ncol(mutationMat)
	if(!is.null(maxSSimu)){
		nSimu <- nrow(maxSSimu)
		maxSize <- ncol(maxSSimu) + 1
	}else{
		maxSSimu <- funMaxSSimu(mutationMat, nSimu = nSimu, nPairStart = nPairStart, maxSize = maxSize, detail = detail)
	}
	resultReal <- funMaxS(mutationMat, nPairStart = nPairStart, maxSize = maxSize, detail = detail)
	maxSMatReal <- resultReal$maxSMat
	maxSReal <- apply(maxSMatReal, 2, max)
	rankMat <- apply(-rbind(maxSReal, maxSSimu), 2, rank, ties.method = "max")
	minRankVecReal <- min(rankMat[1, ] - 1)
	minRankVecSimu <- apply(apply(-maxSSimu, 2, rank, ties.method = "max"), 1, min)
	p <- mean(minRankVecSimu <= minRankVecReal)
	return(list(p = p, q = minRankVecSimu / nSimu, maxSReal = maxSReal, maxSSimu = maxSSimu, startPairIdx = resultReal$startCombn, startPairS = resultReal$startPairS))
}

funMaxS <- function(mutationMat, nPairStart = 10, maxSize = 6, detail = TRUE){
	geneAll <- colnames(mutationMat)
	M <- ncol(mutationMat)
	tempCombn <- combn(M, 2)
	nPair <- ncol(tempCombn)
	pairS <- rep(NA, nPair); names(pairS) <- 1:nPair
	for(i in 1:nPair){
		pairS[i] <- funEstimate(mutationMat[, tempCombn[, i]])$S
		if(detail)cat(i, "\t", geneAll[tempCombn[, i]], pairS[i], "\n")
	}
	startPairS <- sort(pairS, decreasing = T)[1:nPairStart]
	startCombn <- tempCombn[, as.integer(names(startPairS))]
	maxSMat <- matrix(NA, nPairStart, maxSize - 1, dimnames = list(1:nPairStart, 2:maxSize))
	geneMat <- matrix(NA, nPairStart, maxSize, dimnames = list(1:nPairStart, 1:maxSize))
	maxSMat[, "2"] <- startPairS
	geneMat[, "1"] <- geneAll[startCombn[1, ]]
	geneMat[, "2"] <- geneAll[startCombn[2, ]]
	for(i in 1:nPairStart){
		tempGeneSet <- geneAll[startCombn[, i]]
		for(j in 2:(maxSize - 1)){
			tempAdd <- funAdd1(tempGeneSet, mutationMat, detail = detail)
			tempGeneSet <- tempAdd$gene
			maxSMat[i, j] <- tempAdd$S
			geneMat[i, j + 1] <- tempGeneSet[j + 1]
		}
	}
	return(list(maxSMat = maxSMat, geneMat = geneMat, startCombn = startCombn, startPairS = startPairS))
}

funMaxSSimu <- function(mutationMat, nSimu = 1, nPairStart = 10, maxSize = 3, detail = TRUE){
	N <- nrow(mutationMat)
	M <- ncol(mutationMat)
	maxSArray <- array(NA, c(nPairStart, maxSize - 1, nSimu))
	for(s in 1:nSimu){
		permutationMat <- mutationMat
		for(j in 1:M)permutationMat[, j] <- mutationMat[sample(N), j]
		maxSArray[, , s] <- funMaxS(permutationMat, nPairStart = nPairStart, maxSize = maxSize, detail = detail)$maxSMat
		if(detail)cat("Simulation", s, "done", "\n")
	}
	maxSSimu <- apply(maxSArray, 3:2, max)
}
