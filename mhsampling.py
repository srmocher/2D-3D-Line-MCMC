import numpy as np
from scipy.stats import multivariate_normal

# common camera matrix for all scenarios
camera_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]])
num_iterations = 50000

def mh_sample_posterior_3d(t_ratios,r_points,sigma_2d,mu_3d,sigma_3d):
    num_points = r_points.shape[0]

    p_i,p_f = multivariate_normal.rvs(mu_3d,sigma_3d,size=2)

    for i in range(0,num_iterations):
        projected_point_pi = camera_matrix.dot(np.append(p_i,[1]))
        projected_point_pf = camera_matrix.dot(np.append(p_f,[1]))
        #q_s = np.zeros(shape=(num_points,2))
        q_i = (projected_point_pi/projected_point_pi[2])[0:2]
        q_f = (projected_point_pf/projected_point_pf[2])[0:2]

        q_s = np.zeros(shape=(num_points,2))
        for j in range(0,num_points):
            q_s[j] = q_i + (q_f-q_i)*t_ratios[j]

        #log_q_s = np.log(q_s)
        points_3d = [p_i,p_f]
        points_2d = r_points
        likelihood_pdf = np.zeros(shape=(num_points,2))
        for j in range(0,num_points):
            likelihood_pdf[j] = multivariate_normal.logpdf(points_2d[j],mean=q_s[j],cov=sigma_2d)
        prior_pdf = multivariate_normal.logpdf(points_3d,mean=mu_3d,cov=sigma_3d)

        # posterior with old values of pf and pi
        log_posterior_3d = np.sum(likelihood_pdf) + np.sum(prior_pdf)

        p_i_next = multivariate_normal.rvs(mean=mu_3d,cov=sigma_3d)
        p_f_next = multivariate_normal.rvs(mean=mu_3d,cov=sigma_3d)

        projected_pi_current = camera_matrix.dot(np.append(p_i_next,[1]))
        projected_pf_current = camera_matrix.dot(np.append(p_f_next,[1]))

        q_i_next = projected_pi_current[0:2]/projected_pi_current[2]
        q_f_next = projected_pf_current[0:2]/projected_pf_current[2]

        q_s_next = np.zeros(shape=(num_points,2))
        for j in range(0,num_points):
            q_s_next[j] = q_i_next + (q_f_next - q_i_next)*t_ratios[j]
        points_3d_next = [p_i_next,p_f_next]

        likelihood_pdf_next = np.zeros(shape=(num_points,2))
        for j in range(0,num_points):
            likelihood_pdf_next[j] = multivariate_normal.logpdf(points_2d[j],mean=q_s_next[j],cov=sigma_2d)
        prior_pdf_next = multivariate_normal.logpdf(points_3d_next,mean=mu_3d,cov=sigma_3d)

        # posterior with new values of pf and pi sampled from the normal distribution
        log_posterior_3d_next = np.sum(prior_pdf_next) + np.sum(likelihood_pdf_next)

        log_alpha = log_posterior_3d_next - log_posterior_3d

        random_uniform_number = np.random.random()

        #accept
        if log_alpha >= np.log(random_uniform_number):
            p_i = p_i_next
            p_f = p_f_next


r_points = np.genfromtxt('data/points_2d_camera_1.csv',delimiter=',')
t_ratios = np.genfromtxt('data/inputs.csv',delimiter=',')
sigma_3d = 6*np.identity(3)
mu_3d = np.array([0,0,4]).T
sigma_2d = (0.05)*(0.05)*np.identity(2)
mh_sample_posterior_3d(t_ratios,r_points,sigma_2d,mu_3d,sigma_3d)


