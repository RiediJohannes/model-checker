use super::aiger::AIG;
use crate::logic::VAR_OFFSET;

pub fn print_sat_model(graph: &AIG, model: &[i8]) {
    let num_i = graph.inputs.len();
    let num_l = graph.latches.len();
    let num_a = graph.and_gates.len();
    let frame_size = num_i + num_l + num_a;

    println!("#--- SAT Model Assignment (True Variables) ---");

    for (idx, &val) in model.iter().enumerate().skip(VAR_OFFSET) {
        // Only print variables that were set to true
        if val != 1 {
            continue;
        }

        let normalized = idx - VAR_OFFSET;
        let t = normalized / frame_size; // Time step
        let f = normalized % frame_size; // Offset in frame

        // Determine the semantic meaning of the variable within the current frame
        let (label, local_idx) = if f < num_i {
            ("Input", f)
        } else if f < num_i + num_l {
            ("Latch", f - num_i)
        } else {
            ("AND", f - (num_i + num_l))
        };

        println!("{:>4}: {}_{}@{}", idx, label, local_idx, t);
    }
    println!("--------------------------------------------");
}

pub fn print_input_trace(graph: &AIG, model: &[i8]) {
    let num_i = graph.inputs.len();

    // The first variable is the constant bottom
    let total_vars = model.len().saturating_sub(1);
    let vars_per_frame = graph.variables().count();
    let num_steps = total_vars / vars_per_frame;

    // Header
    println!("\n=== Input Trace (Counter-Example) ===");
    print!("{:>4} | ", "t");
    for i in 0..num_i {
        print!("In_{:<2} ", i);
    }
    println!("\n{:-<5}+{:-<}", "", "-".repeat(num_i * 5));

    for k in 0..num_steps {
        print!("{:>4} | ", k);
        for i in 0..num_i {
            // Use frame_size derived from variables().len()
            let idx = VAR_OFFSET + (k * vars_per_frame) + i;
            if let Some(&val) = model.get(idx) {
                let display = match val {
                    1  => " T  ", // True
                    -1 => " F  ", // False
                    0  => " x  ", // Undefined
                    _  => " !  ", // Should not happen
                };
                print!("{}", display);
            }
        }
        println!();
    }
    println!("=====================================\n");
}