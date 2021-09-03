use csv;
use ndarray::{self, Array1, Array2, ArrayView1, Axis};
use rand;

// specific csv read function. this assumes the format:
// id, val1, val2, class
// and ignores id, class
fn read_csv(fname: String) -> Array2<f64> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b';')
        .from_path(fname)
        .unwrap();

    println!("Reading csv...");
    // let cols = reader.headers().unwrap().len();
    let rows = reader.records().count();
    println!("Found {} columns and {} rows.", rows, 2);

    // move to beginning of file
    let pos = csv::Position::new();
    reader.seek(pos).unwrap();

    let mut out_vals = Array2::<f64>::zeros((rows, 2));
    for (record, mut out) in reader.records().skip(1).zip(out_vals.outer_iter_mut()) {
        let cur_record: csv::StringRecord = record.unwrap();
        let c1: f64 = cur_record.get(0).unwrap().parse().unwrap();
        let c2: f64 = cur_record.get(1).unwrap().parse().unwrap();
        out[0] = c1;
        out[1] = c2;
    }
    out_vals
}

fn to_record(vals: &ArrayView1<f64>) -> csv::StringRecord {
    let mut record = csv::StringRecord::with_capacity(10, vals.dim());
    for val in vals.iter() {
        record.push_field(&format!("{}", val).as_str());
    }
    record
}

fn to_csv(arr: Array2<f64>, fname: String) {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b';')
        .from_path(fname)
        .unwrap();

    for row in arr.outer_iter() {
        writer.write_record(&to_record(&row)).unwrap();
    }
    writer.flush().unwrap();
}

fn dist_sq(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    (a.into_owned() - b.into_owned()).map(|val| val * val).sum()
}

/// compute fuzzy memberships for elements of data to clusters in cluster.
/// write membership to memberships
/// q: fuzzifier
/// data: data to compute memberships for
/// clusters: clusters
/// memberships: write membership information here
fn compute_memberships(
    q: f64,
    data: &Array2<f64>,
    clusters: &Array2<f64>,
    memberships: &mut Array2<f64>,
) {
    // Membrships are distances for now
    for (i, val) in data.outer_iter().enumerate() {
        for (j, cluster) in clusters.outer_iter().enumerate() {
            memberships[[i, j]] = dist_sq(val, cluster).powf(1.0 / (1.0 - q));
        }
    }

    // compute cluster memberships
    let inv_dist_sums = memberships.sum_axis(Axis(1)).map(|f| 1.0 / f);
    for (mut cluster_dists, inv_dist_sum) in memberships.outer_iter_mut().zip(inv_dist_sums.iter())
    {
        cluster_dists.mapv_inplace(|val: f64| val * inv_dist_sum);
    }
}

fn compute_nearest(data: &Array2<f64>, clusters: &Array2<f64>) -> Array2<f64> {
    let mut out = data.clone();
    let mut nearest_clusters = Array1::<f64>::zeros(data.dim().0);

    for (val, nearest) in data.outer_iter().zip(nearest_clusters.iter_mut()) {
        let mut min_cluster: usize = 0;
        let mut min_dist = std::f64::MAX;
        for (j, cluster) in clusters.outer_iter().enumerate() {
            let cur_dist = dist_sq(val, cluster);
            if cur_dist < min_dist {
                min_dist = cur_dist;
                min_cluster = j;
            }
        }
        *nearest = min_cluster as f64;
    }

    out.push_column(nearest_clusters.view()).unwrap();
    out
}

/// k: number of clusters
/// n_iter: number of iterations to perform
/// q: fuzzifier
/// data: data to cluster (rows are data points)
fn cluster_k_means_fuzzy(k: usize, n_iter: usize, q: f64, data: &Array2<f64>) -> Array2<f64> {
    let size = data.dim();

    // cluster initialisation as random between 0 and 1
    println!("Initialised cluster positions to:");
    let mut clusters = Array2::<f64>::zeros((k, size.1));
    for mut cluster in clusters.outer_iter_mut() {
        for i in 0..size.1 {
            let val: f64 = rand::random();
            cluster[i] = val * 2.0 - 1.0;
        }
        println!("{:?}", cluster);
    }

    for _ in 0..n_iter {
        let mut memberships = Array2::<f64>::zeros((size.0, k));
        compute_memberships(q, &data, &clusters, &mut memberships);

        println!("DATA:");
        println!("{:?}", data);
        println!("MEMBERSHIPS:");
        println!("{:?}", memberships);
        println!("CLUSTERS");
        println!("{:?}", clusters);
        println!(
            "{:?}, {:?}",
            memberships.axis_iter(Axis(1)).len(),
            clusters.axis_iter(Axis(0)).len()
        );
        // compute new cluster means
        for (mut cluster, membership) in clusters
            .axis_iter_mut(Axis(0))
            .zip(memberships.axis_iter(Axis(1)))
        {
            let mem_sums = membership.mapv(|val: f64| val.powf(q)).sum();
            let fac = membership.mapv(|val| val.powf(q) / mem_sums);
            println!("{:?}", fac);
            cluster.assign(&fac.dot(data));
        }
    }

    clusters
}

// TODO: command line parameters, use command line input
fn main() {
    let n_clusters: usize = 3;
    let n_iter: usize = 10;
    let fuzzifier: f64 = 2.0;
    let input_vals = read_csv(String::from("files/sample_data.csv"));
    let clusters = cluster_k_means_fuzzy(n_clusters, n_iter, fuzzifier, &input_vals);
    let out_vals = compute_nearest(&input_vals, &clusters);
    let mut memberships = Array2::<f64>::zeros((out_vals.dim().0, n_clusters));
    compute_memberships(fuzzifier, &input_vals, &clusters, &mut memberships);
    to_csv(out_vals, String::from("files/predicted_classes.csv"));
    to_csv(clusters, String::from("files/clusters.csv"));
    to_csv(memberships, String::from("files/memberships.csv"));
}
