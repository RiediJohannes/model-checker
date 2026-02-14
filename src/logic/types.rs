#![allow(clippy::upper_case_acronyms)]

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
impl<const N: usize> From<[Literal; N]> for Clause {
    fn from(arr: [Literal; N]) -> Self {
        Self::new(arr)
    }
}
impl From<Literal> for Clause {
    fn from(lit: Literal) -> Self {
        Self::new([lit])
    }
}
impl Debug for Clause {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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
impl<C> FromIterator<C> for CNF
where
    C: Into<Clause>
{
    fn from_iter<T: IntoIterator<Item = C>>(iter: T) -> Self {
        let clauses = iter.into_iter().map(|c| c.into()).collect();
        Self { clauses }
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
#[allow(clippy::suspicious_arithmetic_impl)]
impl BitAnd<&Clause> for CNF {
    type Output = CNF;
    fn bitand(self, rhs: &Clause) -> Self::Output {
        let mut merged_vec = Vec::with_capacity(self.clauses.len() + 1);
        merged_vec.extend_from_slice(&self.clauses);
        merged_vec.push(rhs.clone());
        CNF::from(merged_vec)
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
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

#[macro_export]
macro_rules! cnf {
    ( $( [ $( $lit:expr ),* ] ),* ) => {
        $crate::logic::CNF::from(vec![
            $( $crate::logic::Clause::from([ $( $lit ),* ]) ),*
        ])
    };
}


/// Short for Extended CNF.</br>
/// Extends a formula in CNF by an output tseitin literal that is true iff the formula is satisfiable.
#[derive(Clone)]
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
impl Debug for XCNF {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "XCNF {{ clauses: {{ {:?} }}, out_lit: {} }}", self.clauses, self.out_lit)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cnf_from_clause() {
        let a = Literal::from_var(1);
        let b = Literal::from_var(2);
        let c = Literal::from_var(3);
        let clause: Clause = Clause::new([a, -b, c]);

        let cnf = CNF::from(clause.clone());
        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.clauses[0].lits.len(), 3);
        assert_eq!(cnf.clauses[0], clause);
    }

    #[test]
    fn cnf_from_unit() {
        let a = Literal::from_var(1);
        let cnf = CNF::from(a);
        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.clauses[0].lits.len(), 1);
        assert_eq!(cnf.clauses[0].lits[0], a);

        let b = Literal::from_var(2);
        let cnf = CNF::from(-b);
        assert_eq!(cnf.clauses.len(), 1);
        assert_eq!(cnf.clauses[0].lits.len(), 1);
        assert_eq!(cnf.clauses[0].lits[0], -b);
    }

    #[test]
    fn xcnf_from_unit() {
        let a = Literal::from_var(1);
        let xcnf = XCNF::from(a);
        assert_eq!(xcnf.clauses.len(), 1);
        assert_eq!(xcnf.clauses[0].lits.len(), 1);
        assert_eq!(xcnf.clauses[0].lits[0], a);
        assert_eq!(xcnf.out_lit, a);

        let b = Literal::from_var(2);
        let xcnf = XCNF::from(-b);
        assert_eq!(xcnf.clauses.len(), 1);
        assert_eq!(xcnf.clauses[0].lits.len(), 1);
        assert_eq!(xcnf.clauses[0].lits[0], -b);
        assert_eq!(xcnf.out_lit, -b);
    }

    #[test]
    fn cnf_and_cnf() {
        // Case 1: Unit clauses
        let a = Literal::from_var(1);
        let b = Literal::from_var(2);
        let c = Literal::from_var(3);

        let left_cnf = cnf![[a, b]];
        let right_cnf = CNF::from(c);

        let result: CNF = &left_cnf & &right_cnf;
        assert_eq!(result.clauses.len(), left_cnf.clauses.len() + right_cnf.clauses.len());
        assert_eq!(result.clauses.len(), 2);
        assert_eq!(&result.clauses[0], &left_cnf.clauses[0]);
        assert_eq!(&result.clauses[1], &right_cnf.clauses[0]);


        // Case 2: More complex clauses
        let left_cnf = cnf![[-a, b]];
        let right_cnf = cnf![[-b, c], [-c, a]];

        let result: CNF = &left_cnf & &right_cnf;
        assert_eq!(result.clauses.len(), left_cnf.clauses.len() + right_cnf.clauses.len());
        assert_eq!(result.clauses.len(), 3);
        assert_eq!(&result.clauses[0], &left_cnf.clauses[0]);
        assert_eq!(&result.clauses[1], &right_cnf.clauses[0]);
        assert_eq!(&result.clauses[2], &right_cnf.clauses[1]);
    }

    #[test]
    fn cnf_and_clause() {
        let a = Literal::from_var(1);
        let b = Literal::from_var(2);
        let c = Literal::from_var(3);

        let cnf = cnf![[a, b], [-b]];
        let clause = Clause::from([a, c]);

        let result: CNF = &cnf & &clause;
        assert_eq!(result.clauses.len(), 3);
        assert_eq!(&result.clauses[0], &cnf.clauses[0]);
        assert_eq!(&result.clauses[1], &cnf.clauses[1]);
        assert_eq!(&result.clauses[2], &clause);
    }

    #[test]
    fn clause_and_clause() {
        let a = Literal::from_var(1);
        let b = Literal::from_var(2);
        let c = Literal::from_var(3);

        let left_clause = Clause::from([-a, b]);
        let right_clause = Clause::from([-b, c]);

        let result: CNF = &left_clause & &right_clause;
        assert_eq!(result.clauses.len(), 2);
        assert_eq!(&result.clauses[0], &left_clause);
        assert_eq!(&result.clauses[1], &right_clause);
    }
}