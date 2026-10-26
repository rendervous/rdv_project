// Shared-memory occupancy/uniformity detection, reusable by any node that
// needs to know whether the whole workgroup (all 32 lanes) is actually
// alive right now *and* cooperating on the same logical instance - a
// Library's own code is embedded exactly once in the whole megakernel
// (unlike a per-node __code__, re-specialized once per distinct generic
// instantiation), so a single plain shared declaration here is enough for
// every caller, with no RDV_CONCAT(RDV_NODE, ...) naming needed to avoid
// collisions.
shared uint _coop_sync_id;
shared uint _coop_sync_uniform;
shared int _coop_sync_count;

// True iff every lane of this workgroup reached this call and passed the
// same my_id - false if the workgroup is only partially occupied (e.g. a
// row count that isn't a multiple of 32, so a tail workgroup has fewer
// than 32 threads actually here - rdv's own dispatch bounds check silently
// skips a node's forward() entirely for threads past the true row count)
// or mixed (rows from more than one instance sharing one workgroup, e.g. a
// batched/offset dispatch).
bool all_thread_sync(uint my_id)
{
    if (gl_LocalInvocationID.x == 0)
    {
        _coop_sync_id = my_id;
        _coop_sync_uniform = 1u;
        _coop_sync_count = 0;
    }
    barrier();
    atomicAdd(_coop_sync_count, 1);
    atomicAnd(_coop_sync_uniform, (my_id == _coop_sync_id) ? 1u : 0u);
    barrier();
    return _coop_sync_count == 32 && _coop_sync_uniform == 1u;
}

void RDV_NODE(uvec3 idx) { } // no per-thread init needed - all_thread_sync self-initializes via lane 0.
