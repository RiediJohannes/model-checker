/*****************************************************************************************[Proof.C]
MiniSat -- Copyright (c) 2003-2005, Niklas Een, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/

#include "Proof.h"
#include "Sort.h"


//=================================================================================================
// Temporary files handling:


class TempFiles {
    vec<cchar*> files;      // For clean-up purposed on abnormal exit.

public:
   ~TempFiles()
    {
        for (int i = 0; i < files.size(); i++)
            remove(files[i]);
            //printf("Didn't delete:\n  %s\n", files[i]);
    }

    // Returns a read-only string with the filename of the temporary file. The pointer can be used to 
    // remove the file (which is otherwise done automatically upon program exit).
    //
    char* open(File& fp)
    {
        char*   name;
        for(;;){
            name = tempnam(NULL, NULL);     // (gcc complains about this... stupid gcc...)
            assert(name != NULL);
            fp.open(name, "wx+");
            if (fp.null())
                xfree(name);
            else{
                files.push(name);
                return name;
            }
        }
    }
};
static TempFiles temp_files;       // (should be singleton)


//=================================================================================================
// Proof logging:


Proof::Proof()
{
    fp_name    = temp_files.open(fp);
    id_counter = 0;
    trav       = NULL;
}


Proof::Proof(ProofTraverser& t)
{
    id_counter = 0;
    trav       = &t;
}


ClauseId Proof::addRoot(vec<Lit>& cl)
{
    cl.copyTo(clause);
    sortUnique(clause);

    if (trav != NULL)
        trav->root(clause);
    if (!fp.null()){
        putUInt(fp, index(clause[0]) << 1);
        for (int i = 1; i < clause.size(); i++)
            putUInt(fp, index(clause[i]) - index(clause[i-1]));
        putUInt(fp, 0);     // (0 is safe terminator since we removed duplicates)
    }

    return id_counter++;
}


void Proof::beginChain(ClauseId start)
{
    assert(start != ClauseId_NULL);
    chain_id .clear();
    chain_var.clear();
    chain_id.push(start);
}


void Proof::resolve(ClauseId next, Var x)
{
    assert(next != ClauseId_NULL);
    chain_id .push(next);
    chain_var.push(x);
}


ClauseId Proof::endChain()
{
    assert(chain_id.size() == chain_var.size() + 1);
    if (chain_id.size() == 1)
        return chain_id[0];
    else{
        if (trav != NULL)
            trav->chain(chain_id, chain_var);
        if (!fp.null()){
            putUInt(fp, ((id_counter - chain_id[0]) << 1) | 1);
            for (int i = 0; i < chain_var.size(); i++)
                putUInt(fp, chain_var[i] + 1),
                putUInt(fp, id_counter - chain_id[i+1]);
            putUInt(fp, 0);
        }

        return id_counter++;
    }
}


void Proof::deleted(ClauseId gone)
{
    if (trav != NULL)
        trav->deleted(gone);
    if (!fp.null()){
        putUInt(fp, ((id_counter - gone) << 1) | 1);
        putUInt(fp, 0);
    }
}


//=================================================================================================
// Read-back methods:


void Proof::compress(Proof& dst, ClauseId goal)
{
    assert(!fp.null());
    assert(false);  // Not yet!    
}


bool Proof::save(cchar* filename)
{
    assert(!fp.null());

    // Switch to read mode:
    fp.setMode(READ);
    fp.seek(0);

    // Copy file:
    File    out(filename, "wox");
    if (out.null())
        return false;

    while (!fp.eof())
        out.putChar(fp.getChar());

    // Restore write (proof-logging) mode:
    fp.seek(0, SEEK_END);
    fp.setMode(WRITE);
    return true;
}


void Proof::traverse(ProofTraverser& trav, ClauseId goal)
{
    assert(!fp.null());

    // Switch to read mode:
    fp.setMode(READ);
    fp.seek(0);

    // Traverse proof:
    if (goal == ClauseId_NULL)
        goal = last();

    uint64  tmp;
    int     idx;
    for(ClauseId id = 0; id <= goal; id++){
        tmp = getUInt(fp);
        if ((tmp & 1) == 0){
            // Root clause:
            clause.clear();
            idx = tmp >> 1;
            clause.push(toLit(idx));
            for(;;){
                tmp = getUInt(fp);
                if (tmp == 0) break;
                idx += tmp;
                clause.push(toLit(idx));
            }
            trav.root(clause);

        }else{
            // Derivation or Deletion:
            chain_id .clear();
            chain_var.clear();
            chain_id.push(id - (tmp >> 1));
            for(;;){
                tmp = getUInt(fp);
                if (tmp == 0) break;
                chain_var.push(tmp - 1);
                tmp = getUInt(fp);
                chain_id.push(id - tmp);
            }

            if (chain_var.size() == 0)
                id--,   // (no new clause introduced)
                trav.deleted(chain_id[0]);
            else
                trav.chain(chain_id, chain_var);
        }
    }
    trav.done();

    // Restore write (proof-logging) mode:
    fp.seek(0, SEEK_END);
    fp.setMode(WRITE);
}


//============ Custom Proof Traverser ================
#include "Sort.h"
#include "model-checker/src/minisat.rs.h"  // import shared types


/// Converts a clause c to a Rust slice of i32
inline rust::Slice<const int32_t> toSlice(const vec<Lit>& c, std::vector<int32_t>& storage) {
    storage.clear();
    storage.reserve(c.size());

    for (int i = 0; i < c.size(); ++i) {
        storage.push_back(index(c[i]));
    }

    return {storage.data(), storage.size()};
}

std::unique_ptr<std::vector<int32_t>> toVec(const vec<Lit>& c) {
    std::unique_ptr<std::vector<int32_t>> out(
        new std::vector<int32_t>()
    );

    out->reserve(c.size());
    for (int i = 0; i < c.size(); ++i) {
        out->push_back(index(c[i]));
    }

    return out;
}

static void resolve(vec<Lit>& main, vec<Lit>& other, Var x)
{
    Lit  p;
    bool ok1 = false, ok2 = false;
    for (int i = 0; i < main.size(); i++){
        if (var(main[i]) == x){
            ok1 = true, p = main[i];
            main[i] = main.last();
            main.pop();
            break;
        }
    }

    for (int i = 0; i < other.size(); i++){
        if (var(other[i]) != x)
            main.push(other[i]);
        else{
            if (p != ~other[i])
                printf("PROOF ERROR! Resolved on variable with SAME polarity in both clauses: %d\n", x+1);
            ok2 = true;
        }
    }

    if (!ok1 || !ok2)
        printf("PROOF ERROR! Resolved on missing variable: %d\n", x+1);

    sortUnique(main);
}

void CallbackTraverser::root(const vec<Lit>& c) {
    printf("ROOT");
    // for (int i = 0; i < c.size(); i++) printf(" %s%d", sign(c[i]) ? "-" : "", var(c[i])+1);
    for (int i = 0; i < c.size(); i++) printf(" %d", index(c[i]));
    printf("\n");

    clauses.push();
    c.copyTo(clauses.last());

    // Notify the resolution proof store of the new clause
    // std::vector<int32_t> raw_lits;
    resolution.notify_clause(clauses.size() - 1, toVec(c));
}

void CallbackTraverser::chain(const vec<ClauseId>& cs, const vec<Var>& xs) {
    printf("CHAIN %d", cs[0]);
    for (int i = 0; i < xs.size(); i++) printf(" [%d] %d", xs[i]+1, cs[i+1]);
    printf("\n");

    clauses.push();
    vec<Lit>& c = clauses.last();
    clauses[cs[0]].copyTo(c);

    for (int i = 0; i < xs.size(); i++) {
        resolve(c, clauses[cs[i+1]], xs[i]);

        // If we are not in the first iteration any more, then the left resolution parent is the previous resolvent
        const int left_parent_id = i == 0 ? cs[0] : resolvent_id;
        // The last resolution step in a chain yields a new clause with a non-negative ID
        // In-between results of resolution chains get negative IDs instead (decrement previous resolvent_id)
        const int result_id = i == xs.size() - 1 ? clauses.size() - 1 : --resolvent_id;

        std::vector<int32_t> raw_lits;
        resolution.notify_resolution(result_id, left_parent_id, cs[i+1], xs[i]+1, toSlice(c, raw_lits));
    }
}

void CallbackTraverser::deleted(ClauseId c) {
    // clauses[c].clear();
}

void CallbackTraverser::done() {
    printf("Done traversing.\n");
}
