#ifndef ALLOCATOR_SOA_BLOCK_H
#define ALLOCATOR_SOA_BLOCK_H

#include "allocator/configuration.h"
#include "allocator/util.h"

// Result of block allocation.
struct BlockAllocationResult {
  __device__ BlockAllocationResult(uint64_t allocation_mask_p,
                                   bool block_full_p)
      : allocation_mask(allocation_mask_p), block_full(block_full_p) {}

  uint64_t allocation_mask;

  // Set to true if this allocation request filled up the block entirely.
  bool block_full;
};

enum DeallocationState : int8_t {
  kBlockNowEmpty,     // Deallocate block.
  kBlockNowActive,    // Activate block.
  kRegularDealloc     // Nothing to do.
};

// A SOA block containing objects.
// T: Base type of the block.
// N_Max: Maximum number of objects per block (regardless of type). Currently
//        fixed at 64.
template<class T, int N_Max>
class SoaBlock {
 public:
  static_assert(N_Max == 64, "Not implemented: Custom N_Max.");

  // N_T: Number of object slots.
  static const int N = T::kBlockSize;

  // Bitmap initializer: N_T bits set to 1.
  static const unsigned long long int kBitmapInitState =
      N == N_Max ? (~0ULL) : ((1ULL << N) - 1);

  // Initializes a new block.
  __DEV__ SoaBlock() {
    assert(reinterpret_cast<uintptr_t>(this) % N_Max == 0);   // Alignment.
    type_id = T::kTypeId;
    __threadfence();  // Initialize bitmap after type_id is visible.
    free_bitmap = kBitmapInitState;
    assert(__popcll(free_bitmap) == N);
  }

  // Constructs an object identifier.
  __DEV__ T* make_pointer(uint8_t index) {
    uintptr_t ptr_as_int = index;
    uintptr_t block_size = N;
    ptr_as_int |= block_size << 48;
    uintptr_t type_id = T::kTypeId;
    ptr_as_int |= type_id << 56;
    uintptr_t block_ptr = reinterpret_cast<uintptr_t>(this);
    assert(block_ptr < (1ULL << 49));   // Only 48 bits used in address space.
    assert((block_ptr & 0x3F) == 0);    // Block is aligned.
    ptr_as_int |= block_ptr;
    return reinterpret_cast<T*>(ptr_as_int);
  }

  // Initializes object iteration bitmap.
  __DEV__ void initialize_iteration() {
    iteration_bitmap = (~free_bitmap) & kBitmapInitState;
  }

  __DEV__ DeallocationState deallocate(int position) {
    unsigned long long int before;
    unsigned long long int mask = 1ULL << position;

    do {
      // successful if: bit was "0" (allocated). Needed because we could be in
      // invalidation check.
      before = atomicOr(&free_bitmap, mask);
    } while ((before & mask) != 0);

    int slots_free_before = __popcll(before);
    if (slots_free_before == 0) {
      return kBlockNowActive;
    } else if (slots_free_before == N - 1) {
      return kBlockNowEmpty;
    } else {
      return kRegularDealloc;
    }
  }

  // Only executed by one thread per warp. Request are already aggregated when
  // reaching this function.
  __DEV__ BlockAllocationResult allocate(int bits_to_allocate) {
    // Allocation bits.
    unsigned long long int selected_bits = 0;
    // Set to true if this allocation filled up the block.
    bool filled_block = false, block_full;
    // Helper variables used inside the loop and in the loop condition.
    unsigned long long int before_update, successful_alloc;

    do {
      // Bit set to 1 if slot is free.
      unsigned int rotation_len = warp_id() % 64;
      unsigned long long int updated_mask = rotl(free_bitmap, rotation_len);

      // If there are not enough free slots, allocate as many as possible.
      int free_slots = __popcll(updated_mask);
      int allocation_size = min(free_slots, bits_to_allocate);
      unsigned long long int newly_selected_bits = 0;

      // Generate bitmask for allocation
      for (int i = 0; i < allocation_size; ++i) {
        int next_bit_pos = __ffsll(updated_mask) - 1;
        assert(next_bit_pos >= 0);
        assert(((1ULL << next_bit_pos) & updated_mask) > 0);
        // Clear bit at position `next_bit_pos` in updated mask.
        updated_mask &= updated_mask - 1;
        // Save location of selected bit.
        int next_bit_pos_unrot = (next_bit_pos - rotation_len) % 64;
        newly_selected_bits |= 1ULL << next_bit_pos_unrot;
      }

      assert(__popcll(newly_selected_bits) == allocation_size);
      // Count the number of bits that were selected but already set to false
      // by another thread.
      before_update = atomicAnd(&free_bitmap, ~newly_selected_bits);
      successful_alloc = newly_selected_bits & before_update;
      bits_to_allocate -= __popcll(successful_alloc);
      selected_bits |= successful_alloc;

      // Block full if at least one slot was allocated and "before update"
      // bit-and "now allocated" indicates that block is full.
      block_full = (before_update & ~successful_alloc) == 0;
      filled_block = successful_alloc > 0 && block_full;

      // Stop loop if no more free bits available in this block or all
      // requested allocations completed successfully.
    } while (bits_to_allocate > 0 && !block_full);

    // At most one thread should indicate that the block filled up.
    return BlockAllocationResult(selected_bits, filled_block);
  }

  __DEV__ int DBG_num_bits() {
    return N;
  }

  __DEV__ int DBG_allocated_bits() {
    return N - __popcll(free_bitmap);
  }

  __DEV__ bool is_slot_allocated(int index) {
    return (free_bitmap & (1ULL << index)) == 0;
  }

  // TODO: Should be private.

 public:
  // Dummy area that may be overridden by zero initialization.
  // Data section begins after kBlockDataSectionOffset bytes.
  // TODO: Do we need this on GPU?
  // TODO: Can this be replaced when using ROSE?
  char initialization_header_[kBlockDataSectionOffset - 3*sizeof(unsigned long long int)];

  // Bitmap of free slots.
  unsigned long long int free_bitmap;

  // A copy of ~free_bitmap. Set before the beginning of an iteration. Does
  // not contain dirty objects.
  unsigned long long int iteration_bitmap;

  // Padding to 8 bytes.
  uint8_t type_id;

  static const int kRawStorageBytes = N*T::kObjectSize;

  // Object size must be multiple of 64 bytes.
  static const int kStorageBytes = ((kRawStorageBytes + N_Max - 1) / N_Max) * N_Max;

  static_assert(N <= N_Max, "Assertion failed: N <= N_Max");

  // Data storage.
  char data_[kStorageBytes];
};

#endif  // ALLOCATOR_SOA_BLOCK_H