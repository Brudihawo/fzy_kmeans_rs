use ndarray::Array2;
use std::collections::BTreeMap;
use std::env;
use term_size;

mod lib;
use lib::algo;
use lib::io::{read_csv, to_csv};

#[derive(Clone)]
enum ArgType {
    FloatingNumber(Option<f64>),
    StringType(Option<String>),
    SizeType(Option<usize>),
}

impl ArgType {
    fn float_from_str(input: String) -> ArgType {
        let res = input.parse();
        match res {
            Ok(val) => ArgType::FloatingNumber(Some(val)),
            _ => ArgType::FloatingNumber(None),
        }
    }

    fn string_from_str(input: String) -> ArgType {
        ArgType::StringType(Some(input))
    }

    fn size_from_str(input: String) -> ArgType {
        let res = input.parse();
        match res {
            Ok(val) => ArgType::SizeType(Some(val)),
            _ => ArgType::SizeType(None),
        }
    }

    fn is_some(&self) -> bool {
        match self {
            ArgType::FloatingNumber(val) => val.is_some(),
            ArgType::StringType(val) => val.is_some(),
            ArgType::SizeType(val) => val.is_some(),
        }
    }

    fn is_none(&self) -> bool {
        !self.is_some()
    }

    fn to_string(&self) -> Result<String, ()> {
        if self.is_some() {
            return Ok(match self {
                ArgType::FloatingNumber(val) => format!("{}", val.unwrap()),
                ArgType::StringType(val) => format!("{}", val.as_ref().unwrap()),
                ArgType::SizeType(val) => format!("{}", val.unwrap()),
            })
        }
        Err(())
    }

    fn get_flt(&self) -> Result<f64, ()>{
        if self.is_none() {
            return Err(())
        }
        match self {
            ArgType::FloatingNumber(val) => Ok(val.unwrap()),
            _ => Err(())
        }
    }

    fn get_str(&self) -> Result<String, ()>{
        if self.is_none() {
            return Err(())
        }
        match self {
            ArgType::StringType(val) => Ok(val.clone().unwrap()),
            _ => Err(())
        }
    }

    fn get_size(&self) -> Result<usize, ()>{
        if self.is_none() {
            return Err(())
        }
        match self {
            ArgType::SizeType(val) => Ok(val.unwrap()),
            _ => Err(())
        }
    }
}

struct CmdlineArgument {
    description: &'static str,
    cmdline_expr: &'static str,
    default: ArgType,
    value: ArgType,
}

impl CmdlineArgument {
    fn print_description_str(
        &self,
        term_width: usize,
        param_width: usize,
        default_width: usize,
    ) -> () {
        if term_width < default_width + param_width {
            panic!("Printing wider than the terminal looks like shit!");
        }
        print!(" {: <1$}", self.cmdline_expr, param_width);
        print!("{: <1$}", self.get_default_str(), default_width);
        let mut cur_len = default_width + param_width;
        for word in self.description.split_whitespace() {
            if cur_len + word.len() > term_width {
                print!("\n{: <1$}", "", default_width + param_width + 1);
                cur_len = default_width + param_width;
            }
            print!("{} ", word);
            cur_len += word.len() + 1;
        }
        print!("\n");
    }

    fn get_default_str(&self) -> String {
        match &self.default {
            ArgType::StringType(strtype) => match strtype {
                Some(str) => format!("\"{}\"", str),
                None => String::from("-"),
            },
            ArgType::FloatingNumber(fnum) => match fnum {
                Some(num) => format!("{}", num),
                None => String::from("-"),
            },
            ArgType::SizeType(snum) => match snum {
                Some(num) => format!("{}", num),
                None => String::from("-"),
            },
        }
    }
}

fn parse_args<'a>(
    args: &'a Vec<String>,
) -> Result<(BTreeMap<String, CmdlineArgument>, bool), (BTreeMap<String, CmdlineArgument>, bool)> {
    let mut conf = BTreeMap::<String, CmdlineArgument>::new();

    conf.insert(
        "-i".to_string(),
        CmdlineArgument {
            description: "Path to input file.",
            cmdline_expr: "-i",
            default: ArgType::StringType(None),
            value: ArgType::StringType(None),
        },
    );
    conf.insert(
        "-o".to_string(),
        CmdlineArgument {
            description: "Path to output file.",
            cmdline_expr: "-o",
            default: ArgType::StringType(Some(String::from("out.csv"))),
            value: ArgType::StringType(None),
        },
    );
    conf.insert(
        "-k".to_string(),
        CmdlineArgument {
            description: "Number of Clusters",
            cmdline_expr: "-k",
            default: ArgType::SizeType(Some(5)),
            value: ArgType::SizeType(None),
        },
    );
    conf.insert(
        "-n".to_string(),
        CmdlineArgument {
            description: "Upper Bound of Iteration number.",
            cmdline_expr: "-n",
            default: ArgType::SizeType(Some(10)),
            value: ArgType::SizeType(None),
        },
    );
    conf.insert(
        "-q".to_string(),
        CmdlineArgument {
            description: "Fuzzyfier constant for membership calculation",
            cmdline_expr: "-q",
            default: ArgType::FloatingNumber(Some(2.0)),
            value: ArgType::FloatingNumber(None),
        },
    );

    for val in conf.values_mut() {
        val.value = val.default.clone();
    }

    for (i, arg) in args.iter().enumerate() {
        if arg == &String::from("-h") || arg == &String::from("--help") {
            return Err((conf, true));
        }
        if i + 1 < args.len() {
            if conf.contains_key(arg) {
                let mut tmp = conf.get_mut(arg).unwrap();
                tmp.value = match tmp.default {
                    ArgType::FloatingNumber(_) => ArgType::float_from_str(args[i + 1].clone()),
                    ArgType::StringType(_) => ArgType::string_from_str(args[i + 1].clone()),
                    ArgType::SizeType(_) => ArgType::size_from_str(args[i + 1].clone()),
                }
            }
        }
    }

    let mut err = false;
    for (_, arg) in conf.iter() {
        if arg.value.is_none() {
            err = true;
        }
    }

    if !err {
        Ok((conf, false))
    } else {
        Err((conf, false))
    }
}

fn print_help(config: BTreeMap<String, CmdlineArgument>) {
    const PARAM_TITLE_STR: &str = "Parameter";
    const H_ITEM_SEP: usize = 2;

    let (term_width, _) = term_size::dimensions().unwrap_or((80, 0));
    let mut param_len: usize = 0;
    let mut descr_len: usize = 0;
    let mut default_len: usize = 0;

    for (key, value) in config.iter() {
        if param_len < key.len() {
            param_len = key.len();
        }
        if default_len < value.get_default_str().len() {
            default_len = value.get_default_str().len();
        }
        if descr_len < value.description.len() {
            descr_len = value.description.len();
        }
    }
    param_len = param_len + H_ITEM_SEP;
    default_len = default_len + H_ITEM_SEP;

    if param_len < PARAM_TITLE_STR.len() {
        param_len = PARAM_TITLE_STR.len() + H_ITEM_SEP;
    }

    println!("USAGE: ");
    println!("Parameters without default values are required parameters.");
    print!(
        " {: <3$}{: <4$}{}\n",
        PARAM_TITLE_STR, "Default", "Description", param_len, default_len
    );
    print!("{:-<1$}\n", "", term_width);
    for (_, value) in config.iter() {
        value.print_description_str(term_width, param_len, default_len);
    }
}

fn main() {
    match parse_args(&env::args().collect()) {
        Err((args, print)) => {
            if print {
                print_help(args);
            } else {
                println!("Missing Parameters");
                for (key, arg) in args {
                    if arg.value.is_none() {
                        println!("Parameter {} needs to be provided", key);
                    }
                }
            }
        }
        Ok((args, _)) => {
            let infname = args["-i"].value.get_str().unwrap();
            let ofname = args["-o"].value.get_str().unwrap();
            let n_iter = args["-n"].value.get_size().unwrap();
            let n_clusters = args["-k"].value.get_size().unwrap();
            let fuzzifier = args["-q"].value.get_flt().unwrap();

            let input_vals: Array2<f64> = read_csv(infname);
            let clusters = algo::cluster_k_means_fuzzy(n_clusters, n_iter, fuzzifier, &input_vals);
            let out_vals = algo::compute_nearest(&input_vals, &clusters);
            let mut memberships = Array2::<f64>::zeros((out_vals.dim().0, n_clusters));
            algo::compute_memberships(fuzzifier, &input_vals, &clusters, &mut memberships);

            to_csv(out_vals, String::from(ofname), b';');
        }
    }
    // let n_clusters: usize = 3;
    // let n_iter: usize = 10;
    // let fuzzifier: f64 = 2.0;
    // let input_vals: Array2<f64> = read_csv(String::from("files/sample_data.csv"));
    // let clusters = algo::cluster_k_means_fuzzy(n_clusters, n_iter, fuzzifier, &input_vals);
    // let out_vals = algo::compute_nearest(&input_vals, &clusters);
    // let mut memberships = Array2::<f64>::zeros((out_vals.dim().0, n_clusters));
    // algo::compute_memberships(fuzzifier, &input_vals, &clusters, &mut memberships);

    // to_csv(out_vals, String::from("files/predicted_classes.csv"), b';');
}
