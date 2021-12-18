#-*-coding:UTF-8 -*-

'''
-Created on Feb 10,2017
-@author:Jiajing Guo

-函数功能说明：

	SolveHomotopy(): 解决以下基追踪降噪（BPDN）问题
		min_x  \lambda_coef ||x||_1 + 1/2*||y-Ax||_2^2

	update_primal(): This function computes the minimum step size in the primal update direction and
					 finds change in the primal or dual support with that step.

'''

from __future__ import division
from numpy import *
import numpy as np
from numpy import linalg as la



def update_primal(out_x):
	global N,gamma_x,z_x,xk_temp,del_x_vec,pk_temp,dk,epsilon_gbl,isNonnegative

	#print 'gamma_x',gamma_x
	
	if not out_x and out_x!=0:
		union_gx = set(gamma_x).union(set(out_x))
	else:
		union_gx = set(gamma_x).union(set(np.array([out_x])))
	
	diff_gx = set(range(N)).difference(union_gx)
	gamma_lc = sort(list(diff_gx))

	if isNonnegative:
		delta1 = float("inf")
	else:
		delta1_constr = (epsilon_gbl-pk_temp[gamma_lc])/(1+dk[gamma_lc])
		delta1_pos_ind = np.array([idx for idx,a in enumerate(delta1_constr) if a>0])
		#print 'delta1_pos_ind',delta1_pos_ind
		#delta1_pos = delta1_constr[delta1_pos_ind]
		#delta1 = min(delta1_pos)
		#i_delta1 = delta1_pos.tolist().index(delta1)
		if len(delta1_pos_ind)==0:
			delta1 = []
			i_delta1 = []
		else:
			delta1 = min(delta1_constr[delta1_pos_ind])
			i_delta1 = delta1_constr[delta1_pos_ind].tolist().index(delta1)

		if not delta1:
			delta1 = float("inf")

	delta2_constr = (epsilon_gbl+pk_temp[gamma_lc])/(1-dk[gamma_lc])
	delta2_pos_ind = np.array([idx for idx,a in enumerate(delta2_constr) if a>0])
	#print 'delta2_pos_ind',delta2_pos_ind
	#delta2_pos = delta2_constr[delta2_pos_ind]
	#delta2 = min(delta2_pos)
	#i_delta2 = delta2_pos.tolist().index(delta2)
	if len(delta2_pos_ind)==0:
		delta2 = []
		i_delta2 = []
	else:
		delta2 = min(delta2_constr[delta2_pos_ind])
		i_delta2 = delta2_constr[delta2_pos_ind].tolist().index(delta2)


	#print i_delta2
	if not delta2:
		delta2 = float("inf")

	if delta1>delta2:
		delta = delta2
		i_delta = np.array([gamma_lc[delta2_pos_ind[i_delta2]]])
		#print '11111'
	else:
		delta = delta1
		i_delta = np.array([gamma_lc[delta1_pos_ind[i_delta1]]])
		#print '22222'

	

	delta3_constr = (-xk_temp[gamma_x]/del_x_vec[gamma_x])
	delta3_pos_index = np.array([idx for idx,a in enumerate(delta3_constr) if a>0])
	#print 'delta3_pos_index',delta3_pos_index
	if len(delta3_pos_index)==0:
		delta3 = []
		i_delta3 = []
	else:
		delta3 = min(delta3_constr[delta3_pos_index])
		i_delta3 = delta3_constr[delta3_pos_index].tolist().index(delta3)
		out_x_index = gamma_x[delta3_pos_index[i_delta3]]

	#print 'epsilon_gbl:',epsilon_gbl
	#print 'dk:',dk[0:30]

	out_x = []
	if delta3 and (delta3>0) and (delta3<=delta):
		delta = delta3
		out_x = out_x_index
		#print '33333'

	#print 'out_x',out_x
	#print 'delta',delta
	return i_delta,delta,out_x







def SolveHomotopy(A,y,lambda_coef,tolerance,stoppingCriterion,maxiter, groundTruth = None):
	global N,gamma_x,z_x,xk_temp,del_x_vec,pk_temp,dk,epsilon_gbl,isNonnegative

	eps = pow(2,-52)
	isNonnegative = False
	verbose = True
	xk_l = []

	K,N = shape(A)

	#对偶符号初始化和原始支持初始化
	z_x = zeros(N)
	gamma_x = []

	#Initial step
	Primal_constrk = -dot(A.T,y)

	if isNonnegative:
		c = min(Primal_constrk)
		i = Primal_constrk.tolist().index(c)
		c = max(-c,0)
	else:
		c = max(abs(Primal_constrk))
		i = abs(Primal_constrk).tolist().index(c)

	epsilon_gbl = c

	nz_x = zeros(N)

	if not xk_l:
		xk_l = zeros(N)
		gamma_xk = np.array([i])
	else:
		gamma_xk = np.array([idx for idx,a in enumerate(abs(xk_l)) if a>eps*10])
		nz_x[gamma_xk] = 1

	f_tmp = pow(la.norm(y-dot(A,xk_l)),2)
	f = epsilon_gbl*sum(abs(xk_l))+1/2*f_tmp
	z_x[gamma_xk] = -np.sign(Primal_constrk[gamma_xk])

	z_xk = z_x

	#循环参数
	Iter = 0 
	out_x = []
	old_delta = 0
	count_delta_stop = 0

	#print 'gamma_xk:',gamma_xk
	#print 'gamma_xk type:',type(gamma_xk)
	#print 'gamma_xk len:',size(gamma_xk)
	AtgxAgx = dot(A[:,gamma_xk].T,A[:,gamma_xk])
	#print 'AtgxAgx:',AtgxAgx
	#print 'AtgxAgx type:',type(AtgxAgx)
	#print 'shape:',shape(AtgxAgx)
	#iAtgxAgx = la.inv(dot(A[:,gamma_xk].T,A[:,gamma_xk]))

	iAtgxAgx = 1/AtgxAgx if size(gamma_xk)==1 else la.inv(AtgxAgx)



	while  Iter<maxiter:
		Iter += 1

		#print 'iAtgxAgx:',iAtgxAgx

		gamma_x = gamma_xk
		z_x =z_xk
		x_k = xk_l

		#更新方向
		del_x = dot(iAtgxAgx,z_x[gamma_x])
		del_x_vec = zeros(N)
		del_x_vec[gamma_x] = del_x

		Asupported = A[:,gamma_x]
		Agdelx = dot(Asupported,del_x)
		dk = dot(A.T,Agdelx)

		#在每次操作时，控制机器精度误差：如下
		pk_temp = Primal_constrk
		Pri_temp = abs(abs(Primal_constrk)-epsilon_gbl)
		gammaL_temp = np.array([idx for idx,a in enumerate(Pri_temp) if a<min(epsilon_gbl,2*eps)])
		pk_temp[gammaL_temp] = np.sign(Primal_constrk[gammaL_temp])*epsilon_gbl

		xk_temp = x_k
		xk_temp[abs(x_k)<2*eps] = 0

		#计算步长
		i_delta,delta,out_x = update_primal(out_x)

		if old_delta<4*eps and delta<4*eps:
			count_delta_stop += 1
			if count_delta_stop>=500:
				if verbose:
					print ('stuck in some corner')
				#print 'break1'
				break
		else:
			count_delta_stop = 0
		old_delta = delta

		xk_l = x_k+delta*del_x_vec
		Primal_constrk = Primal_constrk+delta*dk
		epsilon_old = epsilon_gbl
		epsilon_gbl = epsilon_gbl-delta

		if epsilon_gbl<=lambda_coef:
			xk_l = x_k+(epsilon_old-lambda_coef)*del_x_vec
			#print 'break2'
			break


		#计算停止标准和终止测试
		keep_going = True
		if delta!=0:
            #keep_going = norm(xk_1-xG)>tolerance;
			keep_going = la.norm(xk_l-groundTruth)>tolerance
            #prev_f = f
			#f_tmp = pow(la.norm(y-dot(Asupported,xk_l[gamma_x])),2)
			#f = lambda_coef*sum(abs(xk_l))+1/2*f_tmp
			#keep_going = (abs((prev_f-f)/prev_f)>tolerance)
            
            

		#if keep_going and la.norm(xk_l-groundTruth)<100*eps:
			#if verbose:
			#	print ('The iteration is stuck.')
			#keep_going = False

		if not keep_going:
			#print 'break3'
			break

		if out_x:
			#If an element is removed from gamma_x

			#print 'jjjjjjjjjjjj'

			len_gamma = np.array([len(gamma_x)])
			#print out_x
			#print 'out_x type:',type(out_x)
			outx_index = np.array([idx for idx,a in enumerate(gamma_x) if a==out_x])
			gamma_x[outx_index] = gamma_x[len_gamma-1]
			gamma_x[len_gamma-1] = out_x
			gamma_x = gamma_x[0:(len_gamma[0]-1)]
			gamma_xk = gamma_x

			rowi = outx_index
			colj = outx_index
			AtgxAgx_ij = AtgxAgx
			temp_row = AtgxAgx_ij[rowi]
			AtgxAgx_ij[rowi] = AtgxAgx_ij[len_gamma-1]
			AtgxAgx_ij[len_gamma-1] = temp_row
			#AtgxAgx_ij[rowi],AtgxAgx_ij[len_gamma-1] = AtgxAgx_ij[len_gamma-1],AtgxAgx_ij[rowi]
			#AtgxAgx_ij[:,colj],AtgxAgx_ij[:,len_gamma-1] = AtgxAgx_ij[:,len_gamma-1],AtgxAgx_ij[:,colj]
			

			temp_col = AtgxAgx_ij[:,colj]
			AtgxAgx_ij[:,colj] = AtgxAgx_ij[:,len_gamma-1]
			AtgxAgx_ij[:,len_gamma-1] = temp_col

			iAtgxAgx_ij = iAtgxAgx
			temp_row = iAtgxAgx_ij[colj]
			iAtgxAgx_ij[colj] = iAtgxAgx_ij[len_gamma-1]
			iAtgxAgx_ij[len_gamma-1] = temp_row
			#iAtgxAgx_ij[colj],iAtgxAgx_ij[len_gamma-1] = iAtgxAgx_ij[len_gamma-1],iAtgxAgx_ij[colj]
			#iAtgxAgx_ij[:,rowi],iAtgxAgx_ij[:,len_gamma-1] = iAtgxAgx_ij[:,len_gamma-1],iAtgxAgx_ij[:,rowi]
			temp_col = iAtgxAgx_ij[:,rowi]
			iAtgxAgx_ij[:,rowi] = iAtgxAgx_ij[:,len_gamma-1]
			iAtgxAgx_ij[:,len_gamma-1] = temp_col

			AtgxAgx = AtgxAgx_ij[0:(len_gamma[0]-1),0:(len_gamma[0]-1)]

			n = len(AtgxAgx_ij)

			#删除列
			Q11 = iAtgxAgx_ij[0:(n-1),0:(n-1)]
			Q12 = iAtgxAgx_ij[0:(n-1),(n-1)]
			Q21 = iAtgxAgx_ij[(n-1),0:(n-1)]
			Q22 = iAtgxAgx_ij[(n-1),(n-1)]
			Q12Q21_Q22 = outer(Q12,(Q21/Q22))
			#print 'Q12Q21_Q22',Q12Q21_Q22
			iAtgxAgx = Q11-Q12Q21_Q22

			xk_l[out_x] = 0

		else:
			#If an element is added to gamma_x
			gamma_xk = np.hstack((gamma_x,i_delta))
			new_x = i_delta

			#print'kkkkkkkkkkkkkkkk'
			#print gamma_x
			#print new_x


			AtgxAnx = dot(A[:,gamma_x].T,A[:,new_x])
			#print 'AtgxAgx',AtgxAgx
			#print 'AtgxAnx',AtgxAnx
			Attmp_1 = np.hstack((AtgxAgx,AtgxAnx))
			Attmp_2 = np.hstack((AtgxAnx.T,dot(A[:,new_x].T,A[:,i_delta])))
			AtgxAgx_mod = np.vstack((Attmp_1,Attmp_2))

			
			#print 'AtgxAgx_mod',AtgxAgx_mod

			AtgxAgx = AtgxAgx_mod

			n = len(AtgxAgx)

			#增加列
			iA11 = iAtgxAgx
			iA11A12 = dot(iA11,AtgxAgx[0:(n-1),(n-1)])
			A21iA11 = dot(AtgxAgx[(n-1),0:(n-1)],iA11)
			S = AtgxAgx[(n-1),(n-1)]-dot(AtgxAgx[(n-1),0:(n-1)],iA11A12)
			Q11_right = outer(iA11A12,(A21iA11/S))

			#print 'Q11_right:',Q11_right

			iAtgxAgx = zeros([n,n])

			iAtgxAgx[0:(n-1),0:(n-1)] = iA11+Q11_right
			iAtgxAgx[0:(n-1),(n-1)] = -iA11A12/S
			iAtgxAgx[(n-1),0:(n-1)] = -A21iA11/S
			iAtgxAgx[(n-1),(n-1)] = 1/S

			xk_l[i_delta] = 0

		z_xk = zeros(N)
		z_xk[gamma_xk] = -np.sign(Primal_constrk[gamma_xk])
		Primal_constrk[gamma_x] = np.sign(Primal_constrk[gamma_x])*epsilon_gbl

	total_iter = Iter
	x_out = xk_l
	return x_out



























