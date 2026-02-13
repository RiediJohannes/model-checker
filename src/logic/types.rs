use std::fmt::{Debug, Display, Formatter};
use std::ops::{BitAnd, Index};
use super::solving::Literal;

#[derive(Clone,PartialEq,Eq)]
pub struct Clause {
    lits: Box<[Literal]>,
}

impl Clause {
    pub fn new<I>(lits: I) -> Self
    where
        I: IntoIterator<Item = Literal>,
    {
        let v: Vec<Literal> = lits.into_iter().collect();
        Clause::from_vec(v)
    }

    #[inline]
    pub fn from_vec(v: Vec<Literal>) -> Self {
        Self { lits: v.into_boxed_slice() }
    }
}
impl Debug for Clause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Clause: [{}]",
               self.lits.iter()
                   .map(|l| format!("{}", l))
                   .collect::<Vec<_>>()
                   .join(", ")
        )
    }
}
impl Display for Clause {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]",
               self.lits.iter()
                   .map(|l| format!("{}", l))
                   .collect::<Vec<_>>()
                   .join(", ")
        )
    }
}
impl<'a> IntoIterator for &'a Clause {
    type Item = Literal;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, Literal>>;

    fn into_iter(self) -> Self::IntoIter {
        self.lits.iter().copied()
    }
}

impl BitAnd<&Clause> for &Clause {
    type Output = CNF;
    fn bitand(self, rhs: &Clause) -> Self::Output {
        CNF::from(vec![self.clone(), rhs.clone()])
    }
}


#[derive(Clone)]
pub struct CNF {
    pub clauses: Vec<Clause>,
}

impl CNF {
    pub fn len(&self) -> usize {
        self.clauses.len()
    }
}
impl Index<usize> for CNF {
    type Output = Clause;
    fn index(&self, index: usize) -> &Self::Output {
        &self.clauses[index]
    }
}
impl PartialEq<Literal> for &CNF {
    fn eq(&self, lit: &Literal) -> bool {
        self.clauses.len() == 1
            && self.clauses[0].lits.len() == 1
            && self.clauses[0].lits[0] == *lit
    }
}
impl From<Vec<Clause>> for CNF {
    fn from(clauses: Vec<Clause>) -> Self {
        Self { clauses }
    }
}
impl From<Literal> for CNF {
    fn from(lit: Literal) -> Self {
        CNF::from(vec![Clause::new([lit])])
    }
}
impl From<Clause> for CNF {
    fn from(clause: Clause) -> Self {
        CNF::from(vec![clause])
    }
}
impl Debug for CNF {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CNF: {{ {} }}",
               self.clauses.iter()
                   .map(|c| format!("{}", c))
                   .collect::<Vec<_>>()
                   .join(", ")
        )
    }
}
impl BitAnd<&Clause> for CNF {
    type Output = CNF;
    fn bitand(self, rhs: &Clause) -> Self::Output {
        let mut merged_vec = Vec::with_capacity(self.clauses.len() + 1);
        merged_vec.extend_from_slice(&self.clauses);
        merged_vec.push(rhs.clone());
        CNF::from(merged_vec)
    }
}
impl BitAnd<&CNF> for CNF {
    type Output = CNF;
    fn bitand(self, rhs: &CNF) -> Self::Output {
        let mut merged_vec = Vec::with_capacity(self.clauses.len() + rhs.clauses.len());
        merged_vec.extend_from_slice(&self.clauses);
        merged_vec.extend_from_slice(&rhs.clauses);
        CNF::from(merged_vec)
    }
}
impl BitAnd<&Clause> for &CNF {
    type Output = CNF;

    fn bitand(self, rhs: &Clause) -> Self::Output {
        self.clone() & rhs
    }
}
impl BitAnd<&CNF> for &CNF {
    type Output = CNF;
    fn bitand(self, rhs: &CNF) -> Self::Output {
        self.clone() & rhs
    }
}


/// Short for Extended CNF.</br>
/// Extends a formula in CNF by an output tseitin literal that is true iff the formula is satisfiable.
#[derive(Debug,Clone)]
pub struct XCNF {
    pub clauses: CNF,
    pub out_lit: Literal
}

impl XCNF {
    pub fn new(clauses: CNF, output_literal: Literal) -> Self {
        Self {
            clauses,
            out_lit: output_literal
        }
    }
}
impl PartialEq<Literal> for &XCNF {
    fn eq(&self, lit: &Literal) -> bool {
        self.clauses.len() == 1
            && self.clauses[0].lits.len() == 1
            && self.clauses[0].lits[0] == *lit
    }
}
impl From<Literal> for XCNF {
    fn from(lit: Literal) -> Self {
        Self::new(vec![Clause::new([lit])].into(), lit)
    }
}