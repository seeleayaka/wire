# Publication scanner memory incident

The first source-only scan passed before models were restored. A later scan
walked the restored publication tree and attempted to decode the 3.45 GB SAM
checkpoint (and other binary/Git objects) as UTF-8. The audit process reached
about 7.1 GB RSS. The final InitialReviewWorker control failed, and an isolated
retry recorded OpenCV OutOfMemoryError while allocating 159,694,848 bytes.

Only the publication-audit processes were terminated. The scanner now accepts
only eligible bounded text/source/license files and excludes binary models,
Git directories and tokenizer gzip. Earlier 19 actual cases are retained and
must not be declared a completed 20-case acceptance until the final control is
rerun with the memory pressure gone and independently recombined and audited.
No thresholds, labels, model weights, old runtime or selection logic changed.
