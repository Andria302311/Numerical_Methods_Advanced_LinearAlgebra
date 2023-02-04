#Reconstruction of a rectangular matrix from its corrsponding square matrix


#You are basically delving into singular value decomposition (SVD). Let A be your rectangular matrix which of size m×n. Let us assume m<n (other way around is also same). Take B1=AAT and B2=ATA. Take eigen decomposition of both. So that, B1=UΛ1UT and B2=VΛ2VT. Now do the following

#Note the non-zero values inside Λ1 and Λ2, are they related?
#Make a m×m diagonal matrix ΛA with its diagonal entries as square roots of diagonal entries of Λ1. Make the m×n block matrix Λ=[ΛA,0] where the zero part is a m×(n−m) zero matrix.
#Now construct the matrix C=UΛVT. What is the relation between A and C?
#Now think about the other direction, m≥n?.
#Finally, read about SVD in any standard textbook on Matrices.