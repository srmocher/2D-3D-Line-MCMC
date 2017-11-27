import numpy as np
from scipy.stats import multivariate_normal

# common camera matrix for all scenarios
camera_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]])
num_iterations = 50000

def mh_sample_posterior_3d(t_ratios,r_points,sigma_2d,mu_3d,sigma_3d):
    num_points = r_points.size
    p_points = np.zeros(shape=(num_points,3))

    p_i,p_f = multivariate_normal.rvs(mu_3d,sigma_3d,size=2)

    for i in range(0,num_iterations):
        projected_point_pi = camera_matrix.dot(np.append(p_i,[1]))
        projected_point_pf = camera_matrix.dot(np.append(p_f,[1]))
        #q_s = np.zeros(shape=(num_points,2))
        q_i = projected_point_pi[0:1]/projected_point_pi[2]
        q_f = projected_point_pf[0:1]/projected_point_pf[2]

        q_s = q_i + np.multiply(q_f-q_i,t_ratios)

        #log_q_s = np.log(q_s)
        log_3d_points = np.log([p_i,p_f])
        log_2d_points = np.log(r_points)
        likelihood_pdf = multivariate_normal.logpdf(log_2d_points,mean=q_s,cov=sigma_3d)
        prior_pdf = multivariate_normal.logpdf(log_3d_points,mean=mu_3d,cov=sigma_3d)

        #get the proposal distribution
        log_posterior_3d = np.log(likelihood_pdf) + np.log(prior_pdf)

        p_i_next = prior_pdf[0]
        p_f_next = prior_pdf[1]

        projected_pi_next = camera_matrix.dot(np.append(p_i_next,[1]))
        projected_pf_next = camera_matrix.dot(np.append(p_f_next,[1]))

        q_i_next = projected_pi_next[0:1]/projected_pi_next[2]
        q_f_next = projected_pf_next[0:1]/projected_pf_next[2]

        q_s_next = q_i_next + np.multiply(q_f_next - q_i_next,t_ratios)
        log_3d_points_next = np.log([p_i_next,p_f_next])
        


