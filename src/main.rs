use ndarray::Array2;

mod lib;
use lib::io::{read_csv, to_csv};
use lib::algo;

// TODO: command line parameters, use command line input
fn main() {
    let n_clusters: usize = 3;
    let n_iter: usize = 10;
    let fuzzifier: f64 = 2.0;
    let input_vals: Array2<f64> = read_csv(String::from("files/sample_data.csv"));
    let clusters = algo::cluster_k_means_fuzzy(n_clusters, n_iter, fuzzifier, &input_vals);
    let out_vals = algo::compute_nearest(&input_vals, &clusters);
    let mut memberships = Array2::<f64>::zeros((out_vals.dim().0, n_clusters));
    algo::compute_memberships(fuzzifier, &input_vals, &clusters, &mut memberships);

    to_csv(out_vals, String::from("files/predicted_classes.csv"), b';');
    to_csv(clusters, String::from("files/clusters.csv"), b';');
    to_csv(memberships, String::from("files/memberships.csv"), b';');
}
