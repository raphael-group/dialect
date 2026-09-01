#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <libproc.h>
#include <limits.h>
#include <mach/vm_prot.h>
#include <signal.h>
#include <spawn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/proc_info.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern int csops(pid_t process_id, unsigned int operation, void *buffer,
                 size_t buffer_bytes);

#ifndef DIALECT_ROLE
#error DIALECT_ROLE must be supplied as clean, marked, s1, or rebuttal
#endif

#define DIALECT_JOIN_INNER(left, right) left##right
#define DIALECT_JOIN(left, right) DIALECT_JOIN_INNER(left, right)
#define DIALECT_STRINGIFY_INNER(value) #value
#define DIALECT_STRINGIFY(value) DIALECT_STRINGIFY_INNER(value)
#define DIALECT_ROLE_SUPPORTED_clean 1
#define DIALECT_ROLE_SUPPORTED_marked 1
#define DIALECT_ROLE_SUPPORTED_s1 1
#define DIALECT_ROLE_SUPPORTED_rebuttal 1
#define DIALECT_ROLE_SUPPORTED(value)                                          \
  DIALECT_JOIN(DIALECT_ROLE_SUPPORTED_, value)

#if DIALECT_ROLE_SUPPORTED_clean != 1 || DIALECT_ROLE_SUPPORTED_marked != 1 || \
    DIALECT_ROLE_SUPPORTED_s1 != 1 || DIALECT_ROLE_SUPPORTED_rebuttal != 1
#error internal DIALECT_ROLE allowlist is malformed
#endif

#if DIALECT_ROLE_SUPPORTED(DIALECT_ROLE) != 1
#error DIALECT_ROLE must be exactly clean, marked, s1, or rebuttal
#endif

#define DIALECT_ROLE_TEXT DIALECT_STRINGIFY(DIALECT_ROLE)

#ifdef __builtin_strcmp
#error __builtin_strcmp must not be supplied as a macro
#endif

enum {
  dialect_role_must_be_supported =
      1 / ((__builtin_strcmp(DIALECT_ROLE_TEXT, "clean") == 0) ||
           (__builtin_strcmp(DIALECT_ROLE_TEXT, "marked") == 0) ||
           (__builtin_strcmp(DIALECT_ROLE_TEXT, "s1") == 0) ||
           (__builtin_strcmp(DIALECT_ROLE_TEXT, "rebuttal") == 0))
};

#ifndef DIALECT_MAX_BUNDLE_BYTES
#error DIALECT_MAX_BUNDLE_BYTES must be supplied by the authority builder
#endif
#if DIALECT_MAX_BUNDLE_BYTES <= 0
#error DIALECT_MAX_BUNDLE_BYTES must be positive
#endif

#ifndef DIALECT_RUNTIME_PATH
#error DIALECT_RUNTIME_PATH must be supplied by the authority builder
#endif
#ifndef DIALECT_RUNTIME_SHA256
#error DIALECT_RUNTIME_SHA256 must be supplied by the authority builder
#endif
#ifndef DIALECT_RUNTIME_CDHASH
#error DIALECT_RUNTIME_CDHASH must be supplied by the authority builder
#endif
enum {
  dialect_runtime_cdhash_must_have_40_hex_characters =
      1 / (sizeof(DIALECT_RUNTIME_CDHASH) == 41U)
};
#ifndef DIALECT_RUNTIME_BYTES
#error DIALECT_RUNTIME_BYTES must be supplied by the authority builder
#endif
#ifndef DIALECT_RUNTIME_MODE
#error DIALECT_RUNTIME_MODE must be supplied by the authority builder
#endif
#ifndef DIALECT_RENDERER_PATH
#error DIALECT_RENDERER_PATH must be supplied by the authority builder
#endif
#ifndef DIALECT_RENDERER_SHA256
#error DIALECT_RENDERER_SHA256 must be supplied by the authority builder
#endif
#ifndef DIALECT_RENDERER_BYTES
#error DIALECT_RENDERER_BYTES must be supplied by the authority builder
#endif
#ifndef DIALECT_RENDERER_MODE
#error DIALECT_RENDERER_MODE must be supplied by the authority builder
#endif

#define DIALECT_PROTOCOL "dialect-pdf-derivation-fd-protocol-v1"
#define DIALECT_IO_CHUNK 65536U
#define DIALECT_ATTESTATION_WAIT_NANOSECONDS 5000000L
#define DIALECT_ATTESTATION_WAIT_ITERATIONS 1000U
#define DIALECT_MAX_EXECUTABLE_REGIONS 64U
#define DIALECT_CS_OPS_STATUS 0U
#define DIALECT_CS_OPS_CDHASH 5U
#define DIALECT_CS_VALID 0x00000001U
#define DIALECT_CS_INVALID_ALLOWED 0x00000020U
#define DIALECT_CS_KILL 0x00000200U
#define DIALECT_CS_KILLED 0x01000000U
#define DIALECT_CS_DEBUGGED 0x10000000U
#define DIALECT_CS_SIGNED 0x20000000U
#define DIALECT_REQUIRED_CS_FLAGS                                              \
  (DIALECT_CS_VALID | DIALECT_CS_KILL | DIALECT_CS_SIGNED)
#define DIALECT_REJECTED_CS_FLAGS                                              \
  (DIALECT_CS_INVALID_ALLOWED | DIALECT_CS_KILLED | DIALECT_CS_DEBUGGED)
#define DIALECT_CDHASH_BYTES 20U
#define DIALECT_RENDERER_BOOTSTRAP                                             \
  "import os,sys\n"                                                            \
  "p=sys.argv[1]\n"                                                            \
  "f=int(sys.argv[2],10)\n"                                                    \
  "n=int(sys.argv[3],10)\n"                                                    \
  "b=bytearray()\n"                                                            \
  "while len(b)<n:\n"                                                          \
  " c=os.read(f,min(65536,n-len(b)))\n"                                        \
  " if not c:raise RuntimeError('short renderer read')\n"                      \
  " b.extend(c)\n"                                                             \
  "if os.read(f,1):raise RuntimeError('long renderer read')\n"                 \
  "os.close(f)\n"                                                              \
  "sys.argv=[p]+sys.argv[4:]\n"                                                \
  "g={'__name__':'__main__','__file__':p,'__package__':None,"                  \
  "'__cached__':None,'__spec__':None,'__loader__':None}\n"                     \
  "exec(compile(bytes(b),p,'exec'),g,g)\n"

struct dialect_sha256 {
  uint32_t state[8];
  uint64_t bit_count;
  unsigned char block[64];
  size_t block_size;
};

static uint32_t rotate_right(uint32_t value, unsigned int count) {
  return (value >> count) | (value << (32U - count));
}

static void sha256_transform(struct dialect_sha256 *context,
                             const unsigned char block[64]) {
  static const uint32_t constants[64] = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
  };
  uint32_t words[64];
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
  uint32_t e;
  uint32_t f;
  uint32_t g;
  uint32_t h;
  size_t index;

  for (index = 0; index < 16U; ++index) {
    const size_t offset = index * 4U;
    words[index] = ((uint32_t)block[offset] << 24U) |
                   ((uint32_t)block[offset + 1U] << 16U) |
                   ((uint32_t)block[offset + 2U] << 8U) |
                   (uint32_t)block[offset + 3U];
  }
  for (index = 16U; index < 64U; ++index) {
    const uint32_t x = words[index - 15U];
    const uint32_t y = words[index - 2U];
    const uint32_t small_zero =
        rotate_right(x, 7U) ^ rotate_right(x, 18U) ^ (x >> 3U);
    const uint32_t small_one =
        rotate_right(y, 17U) ^ rotate_right(y, 19U) ^ (y >> 10U);
    words[index] =
        words[index - 16U] + small_zero + words[index - 7U] + small_one;
  }

  a = context->state[0];
  b = context->state[1];
  c = context->state[2];
  d = context->state[3];
  e = context->state[4];
  f = context->state[5];
  g = context->state[6];
  h = context->state[7];
  for (index = 0; index < 64U; ++index) {
    const uint32_t big_one =
        rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
    const uint32_t choose = (e & f) ^ ((~e) & g);
    const uint32_t temporary_one =
        h + big_one + choose + constants[index] + words[index];
    const uint32_t big_zero =
        rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temporary_two = big_zero + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary_one;
    d = c;
    c = b;
    b = a;
    a = temporary_one + temporary_two;
  }
  context->state[0] += a;
  context->state[1] += b;
  context->state[2] += c;
  context->state[3] += d;
  context->state[4] += e;
  context->state[5] += f;
  context->state[6] += g;
  context->state[7] += h;
}

static void sha256_init(struct dialect_sha256 *context) {
  static const uint32_t initial[8] = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };
  memcpy(context->state, initial, sizeof(initial));
  context->bit_count = 0U;
  context->block_size = 0U;
}

static void sha256_update(struct dialect_sha256 *context,
                          const unsigned char *bytes, size_t length) {
  size_t offset = 0U;
  context->bit_count += (uint64_t)length * 8U;
  while (offset < length) {
    size_t available = 64U - context->block_size;
    size_t remaining = length - offset;
    size_t count = remaining < available ? remaining : available;
    memcpy(context->block + context->block_size, bytes + offset, count);
    context->block_size += count;
    offset += count;
    if (context->block_size == 64U) {
      sha256_transform(context, context->block);
      context->block_size = 0U;
    }
  }
}

static void sha256_final(struct dialect_sha256 *context,
                         unsigned char digest[32]) {
  size_t index;
  context->block[context->block_size++] = 0x80U;
  if (context->block_size > 56U) {
    while (context->block_size < 64U) {
      context->block[context->block_size++] = 0U;
    }
    sha256_transform(context, context->block);
    context->block_size = 0U;
  }
  while (context->block_size < 56U) {
    context->block[context->block_size++] = 0U;
  }
  for (index = 0U; index < 8U; ++index) {
    context->block[63U - index] =
        (unsigned char)(context->bit_count >> (index * 8U));
  }
  sha256_transform(context, context->block);
  for (index = 0U; index < 8U; ++index) {
    digest[index * 4U] = (unsigned char)(context->state[index] >> 24U);
    digest[index * 4U + 1U] = (unsigned char)(context->state[index] >> 16U);
    digest[index * 4U + 2U] = (unsigned char)(context->state[index] >> 8U);
    digest[index * 4U + 3U] = (unsigned char)context->state[index];
  }
}

static int hex_value(char value) {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  return -1;
}

static int digest_matches(const unsigned char digest[32],
                          const char *expected) {
  size_t index;
  if (strlen(expected) != 64U) {
    return 0;
  }
  for (index = 0U; index < 32U; ++index) {
    const int high = hex_value(expected[index * 2U]);
    const int low = hex_value(expected[index * 2U + 1U]);
    if (high < 0 || low < 0 ||
        digest[index] != (unsigned char)((high << 4) | low)) {
      return 0;
    }
  }
  return 1;
}

static int stat_identity_matches(const struct stat *first,
                                 const struct stat *second) {
  return first->st_dev == second->st_dev && first->st_ino == second->st_ino &&
         first->st_size == second->st_size &&
         first->st_mtimespec.tv_sec == second->st_mtimespec.tv_sec &&
         first->st_mtimespec.tv_nsec == second->st_mtimespec.tv_nsec &&
         first->st_mode == second->st_mode &&
         first->st_nlink == second->st_nlink && first->st_uid == second->st_uid;
}

static int verify_dependency_descriptor(int descriptor,
                                        const char *expected_sha256,
                                        off_t expected_size,
                                        mode_t expected_mode, int executable,
                                        const struct stat *expected_identity,
                                        struct stat *observed_identity) {
  unsigned char buffer[DIALECT_IO_CHUNK];
  unsigned char digest[32];
  struct dialect_sha256 hash;
  struct stat before;
  struct stat after;
  off_t offset = 0;
  if (fstat(descriptor, &before) != 0) {
    return 0;
  }
  if (!S_ISREG(before.st_mode) || before.st_nlink != 1 ||
      before.st_uid != geteuid() || before.st_size != expected_size ||
      (before.st_mode & 07777) != expected_mode ||
      (executable != 0 && (before.st_mode & S_IXUSR) == 0) ||
      (expected_identity != NULL &&
       !stat_identity_matches(expected_identity, &before))) {
    return 0;
  }
  sha256_init(&hash);
  while (offset < before.st_size) {
    size_t wanted = (size_t)(before.st_size - offset);
    ssize_t count;
    if (wanted > sizeof(buffer)) {
      wanted = sizeof(buffer);
    }
    count = pread(descriptor, buffer, wanted, offset);
    if (count <= 0) {
      return 0;
    }
    sha256_update(&hash, buffer, (size_t)count);
    offset += count;
  }
  sha256_final(&hash, digest);
  if (fstat(descriptor, &after) != 0 ||
      !stat_identity_matches(&before, &after) ||
      !digest_matches(digest, expected_sha256)) {
    return 0;
  }
  if (observed_identity != NULL) {
    *observed_identity = after;
  }
  return 1;
}

static int pin_dependency(const char *path, const char *expected_sha256,
                          off_t expected_size, mode_t expected_mode,
                          int executable, int *descriptor,
                          struct stat *identity) {
  int opened = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (opened < 0 || !verify_dependency_descriptor(opened, expected_sha256,
                                                  expected_size, expected_mode,
                                                  executable, NULL, identity)) {
    if (opened >= 0) {
      (void)close(opened);
    }
    return 0;
  }
  *descriptor = opened;
  return 1;
}

static int parse_source_descriptor(const char *raw, int *descriptor) {
  char *end = NULL;
  long parsed;
  size_t index;

  if (raw == NULL || raw[0] < '1' || raw[0] > '9') {
    return 0;
  }
  for (index = 0U; raw[index] != '\0'; ++index) {
    if (raw[index] < '0' || raw[index] > '9' || index >= 10U) {
      return 0;
    }
  }
  if (index == 0U) {
    return 0;
  }
  errno = 0;
  parsed = strtol(raw, &end, 10);
  if (errno != 0 || end == NULL || *end != '\0' || parsed < 3L ||
      parsed > INT_MAX) {
    return 0;
  }
  *descriptor = (int)parsed;
  return 1;
}

static int verify_source_descriptor(int descriptor) {
  struct stat before;
  struct stat after;
  int descriptor_flags;
  int status_flags;

  status_flags = fcntl(descriptor, F_GETFL);
  descriptor_flags = fcntl(descriptor, F_GETFD);
  if (status_flags < 0 || descriptor_flags < 0 ||
      (status_flags & O_ACCMODE) != O_RDONLY ||
      (status_flags & ~(O_ACCMODE | O_NONBLOCK)) != 0 ||
      fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_nlink != 1 || before.st_uid != geteuid() ||
      before.st_size < 1 || before.st_size > DIALECT_MAX_BUNDLE_BYTES ||
      (before.st_mode & 07777) != 0400 || lseek(descriptor, 0, SEEK_CUR) < 0 ||
      lseek(descriptor, 0, SEEK_SET) != 0 || fstat(descriptor, &after) != 0 ||
      !stat_identity_matches(&before, &after)) {
    return 0;
  }
  return 1;
}

static int descriptor_is_preserved(int descriptor,
                                   const int *preserved_descriptors,
                                   size_t preserved_count) {
  size_t index;
  for (index = 0U; index < preserved_count; ++index) {
    if (descriptor == preserved_descriptors[index]) {
      return 1;
    }
  }
  return 0;
}

static int close_unexpected_descriptors(const int *preserved_descriptors,
                                        size_t preserved_count) {
  DIR *directory;
  struct dirent *entry;
  int directory_descriptor;
  int unexpected[64];
  size_t count = 0U;
  size_t index;

  directory = opendir("/dev/fd");
  if (directory == NULL) {
    return 0;
  }
  directory_descriptor = dirfd(directory);
  if (directory_descriptor < 0) {
    (void)closedir(directory);
    return 0;
  }
  errno = 0;
  while ((entry = readdir(directory)) != NULL) {
    int descriptor;
    if (strcmp(entry->d_name, "0") == 0 || strcmp(entry->d_name, "1") == 0 ||
        strcmp(entry->d_name, "2") == 0) {
      continue;
    }
    if (!parse_source_descriptor(entry->d_name, &descriptor)) {
      if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
        (void)closedir(directory);
        return 0;
      }
      continue;
    }
    if (descriptor > 2 && descriptor != directory_descriptor &&
        !descriptor_is_preserved(descriptor, preserved_descriptors,
                                 preserved_count)) {
      if (count >= sizeof(unexpected) / sizeof(unexpected[0])) {
        (void)closedir(directory);
        return 0;
      }
      unexpected[count++] = descriptor;
    }
  }
  if (errno != 0 || closedir(directory) != 0) {
    return 0;
  }
  for (index = 0U; index < count; ++index) {
    if (close(unexpected[index]) != 0 && errno != EBADF) {
      return 0;
    }
  }
  return 1;
}

static int format_nonnegative_decimal(off_t value, char *buffer,
                                      size_t buffer_bytes) {
  char reverse[32];
  uint64_t remaining;
  size_t count = 0U;
  size_t index;

  if (value < 0 || buffer_bytes == 0U) {
    return 0;
  }
  remaining = (uint64_t)value;
  do {
    if (count >= sizeof(reverse)) {
      return 0;
    }
    reverse[count++] = (char)('0' + (remaining % 10U));
    remaining /= 10U;
  } while (remaining != 0U);
  if (count >= buffer_bytes) {
    return 0;
  }
  for (index = 0U; index < count; ++index) {
    buffer[index] = reverse[count - index - 1U];
  }
  buffer[count] = '\0';
  return 1;
}

struct dialect_handoff {
  char renderer_bytes_text[32];
  char renderer_descriptor_text[32];
  char *arguments[18];
  char *environment[4];
};

static int prepare_verified_handoff(int source_descriptor,
                                    int runtime_descriptor,
                                    const struct stat *runtime_identity,
                                    int renderer_descriptor,
                                    const struct stat *renderer_identity,
                                    char *launcher_argv[],
                                    struct dialect_handoff *handoff) {
  int preserved_descriptors[3];

  preserved_descriptors[0] = source_descriptor;
  preserved_descriptors[1] = runtime_descriptor;
  preserved_descriptors[2] = renderer_descriptor;
  if (!close_unexpected_descriptors(preserved_descriptors,
                                    sizeof(preserved_descriptors) /
                                        sizeof(preserved_descriptors[0])) ||
      !format_nonnegative_decimal((off_t)renderer_descriptor,
                                  handoff->renderer_descriptor_text,
                                  sizeof(handoff->renderer_descriptor_text)) ||
      !format_nonnegative_decimal((off_t)DIALECT_RENDERER_BYTES,
                                  handoff->renderer_bytes_text,
                                  sizeof(handoff->renderer_bytes_text))) {
    return 69;
  }
  if (!verify_source_descriptor(source_descriptor)) {
    return 65;
  }
  if (!verify_dependency_descriptor(
          renderer_descriptor, DIALECT_RENDERER_SHA256,
          (off_t)DIALECT_RENDERER_BYTES, (mode_t)DIALECT_RENDERER_MODE, 0,
          renderer_identity, NULL)) {
    return 68;
  }
  if (!verify_dependency_descriptor(runtime_descriptor, DIALECT_RUNTIME_SHA256,
                                    (off_t)DIALECT_RUNTIME_BYTES,
                                    (mode_t)DIALECT_RUNTIME_MODE, 1,
                                    runtime_identity, NULL)) {
    return 67;
  }

  handoff->arguments[0] = (char *)DIALECT_RUNTIME_PATH;
  handoff->arguments[1] = (char *)"-I";
  handoff->arguments[2] = (char *)"-S";
  handoff->arguments[3] = (char *)"-B";
  handoff->arguments[4] = (char *)"-c";
  handoff->arguments[5] = (char *)DIALECT_RENDERER_BOOTSTRAP;
  handoff->arguments[6] = (char *)DIALECT_RENDERER_PATH;
  handoff->arguments[7] = handoff->renderer_descriptor_text;
  handoff->arguments[8] = handoff->renderer_bytes_text;
  handoff->arguments[9] = launcher_argv[1];
  handoff->arguments[10] = launcher_argv[2];
  handoff->arguments[11] = launcher_argv[3];
  handoff->arguments[12] = launcher_argv[4];
  handoff->arguments[13] = launcher_argv[5];
  handoff->arguments[14] = launcher_argv[6];
  handoff->arguments[15] = launcher_argv[7];
  handoff->arguments[16] = launcher_argv[8];
  handoff->arguments[17] = NULL;
  handoff->environment[0] = (char *)"LANG=C";
  handoff->environment[1] = (char *)"LC_ALL=C";
  handoff->environment[2] = (char *)"TZ=UTC";
  handoff->environment[3] = NULL;
  return 0;
}

static int expected_cdhash(unsigned char output[DIALECT_CDHASH_BYTES]) {
  const volatile char *encoded = DIALECT_RUNTIME_CDHASH;
  size_t index;
  for (index = 0U; index < DIALECT_CDHASH_BYTES; ++index) {
    const int high = hex_value(encoded[index * 2U]);
    const int low = hex_value(encoded[index * 2U + 1U]);
    if (high < 0 || low < 0) {
      return 0;
    }
    output[index] = (unsigned char)((high << 4) | low);
  }
  return 1;
}

static int wait_for_suspended_child(pid_t process_id) {
  struct timespec delay;
  unsigned int iteration;

  delay.tv_sec = 0;
  delay.tv_nsec = DIALECT_ATTESTATION_WAIT_NANOSECONDS;
  for (iteration = 0U; iteration < DIALECT_ATTESTATION_WAIT_ITERATIONS;
       ++iteration) {
    siginfo_t information;
    memset(&information, 0, sizeof(information));
    if (waitid(P_PID, (id_t)process_id, &information,
               WSTOPPED | WEXITED | WNOHANG | WNOWAIT) != 0) {
      if (errno == EINTR) {
        continue;
      }
      return 0;
    }
    if (information.si_pid == process_id) {
      return information.si_signo == SIGCHLD &&
             information.si_code == CLD_STOPPED && information.si_status == 0;
    }
    if (information.si_pid != 0 || nanosleep(&delay, NULL) != 0) {
      return 0;
    }
  }
  return 0;
}

static int darwin_attestation_abi_supported(void) {
  return POSIX_SPAWN_START_SUSPENDED == 0x0080 &&
         POSIX_SPAWN_CLOEXEC_DEFAULT == 0x4000 && P_PID == 1 &&
         WNOHANG == 0x00000001 && WEXITED == 0x00000004 &&
         WSTOPPED == 0x00000008 && WNOWAIT == 0x00000020 && CLD_STOPPED == 5 &&
         VM_PROT_EXECUTE == 0x04 && PROC_PIDREGIONPATHINFO == 8 &&
         sizeof(struct proc_regioninfo) == 96U &&
         offsetof(struct proc_regioninfo, pri_protection) == 0U &&
         offsetof(struct proc_regioninfo, pri_offset) == 16U &&
         offsetof(struct proc_regioninfo, pri_address) == 80U &&
         offsetof(struct proc_regioninfo, pri_size) == 88U &&
         sizeof(struct vinfo_stat) == 136U &&
         offsetof(struct vinfo_stat, vst_dev) == 0U &&
         offsetof(struct vinfo_stat, vst_mode) == 4U &&
         offsetof(struct vinfo_stat, vst_nlink) == 6U &&
         offsetof(struct vinfo_stat, vst_ino) == 8U &&
         offsetof(struct vinfo_stat, vst_uid) == 16U &&
         offsetof(struct vinfo_stat, vst_size) == 88U &&
         sizeof(struct vnode_info) == 152U &&
         sizeof(struct vnode_info_path) == 1176U &&
         offsetof(struct vnode_info_path, vip_path) == 152U &&
         sizeof(struct proc_regionwithpathinfo) == 1272U &&
         offsetof(struct proc_regionwithpathinfo, prp_vip) == 96U;
}

static int mapped_runtime_matches(pid_t process_id,
                                  const struct stat *runtime_identity) {
  uint64_t address = 0U;
  unsigned int iteration;

  if ((dev_t)(uint32_t)runtime_identity->st_dev != runtime_identity->st_dev) {
    return 0;
  }
  for (iteration = 0U; iteration < DIALECT_MAX_EXECUTABLE_REGIONS;
       ++iteration) {
    struct proc_regionwithpathinfo region;
    const struct proc_regioninfo *mapping = &region.prp_prinfo;
    const struct vinfo_stat *vnode = &region.prp_vip.vip_vi.vi_stat;
    int returned;

    memset(&region, 0, sizeof(region));
    returned = proc_pidinfo(process_id, PROC_PIDREGIONPATHINFO, address,
                            &region, (int)sizeof(region));
    if (returned != (int)sizeof(region)) {
      return 0;
    }
    if ((mapping->pri_protection & VM_PROT_EXECUTE) != 0 &&
        mapping->pri_offset == 0U &&
        vnode->vst_dev == (uint32_t)runtime_identity->st_dev &&
        vnode->vst_ino == (uint64_t)runtime_identity->st_ino) {
      return vnode->vst_mode == (uint16_t)runtime_identity->st_mode &&
             vnode->vst_nlink == (uint16_t)runtime_identity->st_nlink &&
             vnode->vst_uid == (uint32_t)runtime_identity->st_uid &&
             vnode->vst_size == (int64_t)runtime_identity->st_size;
    }
    if (mapping->pri_size == 0U ||
        mapping->pri_address > UINT64_MAX - mapping->pri_size ||
        mapping->pri_address + mapping->pri_size <= address) {
      return 0;
    }
    address = mapping->pri_address + mapping->pri_size;
  }
  return 0;
}

static int signing_identity_matches(pid_t process_id) {
  unsigned char expected[DIALECT_CDHASH_BYTES];
  unsigned char observed[DIALECT_CDHASH_BYTES];
  uint32_t status = 0U;

  if (!expected_cdhash(expected) ||
      csops(process_id, DIALECT_CS_OPS_STATUS, &status, sizeof(status)) != 0 ||
      csops(process_id, DIALECT_CS_OPS_CDHASH, observed, sizeof(observed)) !=
          0 ||
      memcmp(expected, observed, sizeof(expected)) != 0 ||
      (status & DIALECT_REQUIRED_CS_FLAGS) != DIALECT_REQUIRED_CS_FLAGS ||
      (status & DIALECT_REJECTED_CS_FLAGS) != 0U) {
    return 0;
  }
  return 1;
}

static int kill_and_reap_child(pid_t process_id) {
  int status;
  int killed;
  pid_t waited;

  do {
    killed = kill(process_id, SIGKILL);
  } while (killed != 0 && errno == EINTR);
  if (killed != 0 && errno != ESRCH) {
    return 0;
  }
  do {
    waited = waitpid(process_id, &status, 0);
  } while (waited < 0 && errno == EINTR);
  return waited == process_id;
}

static int reject_suspended_child(pid_t process_id, int failure_code) {
  return kill_and_reap_child(process_id) ? failure_code : 67;
}

static int child_exit_code(pid_t process_id) {
  int status;
  pid_t waited;

  do {
    waited = waitpid(process_id, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != process_id) {
    return 126;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    const int signal_number = WTERMSIG(status);
    struct sigaction disposition;
    sigset_t unblocked;

    if (signal_number != SIGKILL && signal_number != SIGSTOP) {
      memset(&disposition, 0, sizeof(disposition));
      disposition.sa_handler = SIG_DFL;
      if (sigemptyset(&disposition.sa_mask) != 0 ||
          sigaction(signal_number, &disposition, NULL) != 0) {
        return 126;
      }
    }
    if (sigemptyset(&unblocked) != 0 ||
        sigaddset(&unblocked, signal_number) != 0 ||
        sigprocmask(SIG_UNBLOCK, &unblocked, NULL) != 0 ||
        kill(getpid(), signal_number) != 0) {
      return 126;
    }
    return 126;
  }
  return 126;
}

/*
 * Suspended-process attestation binds the pathname winner to the held runtime
 * vnode before this launcher resumes it. It assumes kernel-enforced suspension
 * is not defeated by an active same-UID peer sending SIGCONT. Dylib and Python
 * stdlib loads, plus same-vnode non-code or post-attestation renderer mutation,
 * remain outside this launcher's containment claim.
 */
static int spawn_attested_runtime(int source_descriptor, int runtime_descriptor,
                                  const struct stat *runtime_identity,
                                  int renderer_descriptor,
                                  const struct stat *renderer_identity,
                                  struct dialect_handoff *handoff) {
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  pid_t process_id = 0;
  int actions_ready = 0;
  int attributes_ready = 0;
  int spawned = 0;
  int status;
  short flags = POSIX_SPAWN_START_SUSPENDED | POSIX_SPAWN_CLOEXEC_DEFAULT;

  if (!darwin_attestation_abi_supported()) {
    return 67;
  }
  status = posix_spawnattr_init(&attributes);
  if (status == 0) {
    attributes_ready = 1;
    status = posix_spawnattr_setflags(&attributes, flags);
  }
  if (status == 0) {
    status = posix_spawn_file_actions_init(&actions);
    if (status == 0) {
      actions_ready = 1;
    }
  }
  if (status == 0) {
    status = posix_spawn_file_actions_addinherit_np(&actions, 0);
  }
  if (status == 0) {
    status = posix_spawn_file_actions_addinherit_np(&actions, 1);
  }
  if (status == 0) {
    status = posix_spawn_file_actions_addinherit_np(&actions, 2);
  }
  if (status == 0) {
    status =
        posix_spawn_file_actions_addinherit_np(&actions, source_descriptor);
  }
  if (status == 0) {
    status =
        posix_spawn_file_actions_addinherit_np(&actions, renderer_descriptor);
  }
  if (status == 0) {
    status = posix_spawn(&process_id, DIALECT_RUNTIME_PATH, &actions,
                         &attributes, handoff->arguments, handoff->environment);
    if (status == 0 && process_id > 0) {
      spawned = 1;
    }
  }
  if (actions_ready != 0 && posix_spawn_file_actions_destroy(&actions) != 0) {
    status = EINVAL;
  }
  if (attributes_ready != 0 && posix_spawnattr_destroy(&attributes) != 0) {
    status = EINVAL;
  }
  if (status != 0 || spawned == 0) {
    if (spawned != 0) {
      return reject_suspended_child(process_id, 67);
    }
    return 67;
  }
  if (!wait_for_suspended_child(process_id)) {
    return reject_suspended_child(process_id, 67);
  }
  if (!verify_source_descriptor(source_descriptor)) {
    return reject_suspended_child(process_id, 65);
  }
  if (!verify_dependency_descriptor(
          renderer_descriptor, DIALECT_RENDERER_SHA256,
          (off_t)DIALECT_RENDERER_BYTES, (mode_t)DIALECT_RENDERER_MODE, 0,
          renderer_identity, NULL)) {
    return reject_suspended_child(process_id, 68);
  }
  if (!verify_dependency_descriptor(runtime_descriptor, DIALECT_RUNTIME_SHA256,
                                    (off_t)DIALECT_RUNTIME_BYTES,
                                    (mode_t)DIALECT_RUNTIME_MODE, 1,
                                    runtime_identity, NULL) ||
      !mapped_runtime_matches(process_id, runtime_identity) ||
      !signing_identity_matches(process_id) ||
      !wait_for_suspended_child(process_id)) {
    return reject_suspended_child(process_id, 67);
  }
  if (kill(process_id, SIGCONT) != 0) {
    return reject_suspended_child(process_id, 67);
  }
  return child_exit_code(process_id);
}

int main(int argc, char *argv[]) {
  int source_descriptor;
  int runtime_descriptor = -1;
  int renderer_descriptor = -1;
  struct stat runtime_identity;
  struct stat renderer_identity;
  struct dialect_handoff handoff;
  int result;

  if (argc != 9 || strcmp(argv[1], "--dialect-derivation-protocol") != 0 ||
      strcmp(argv[2], DIALECT_PROTOCOL) != 0 ||
      strcmp(argv[3], "--pdf-id") != 0 ||
      strcmp(argv[4], DIALECT_ROLE_TEXT) != 0 ||
      strcmp(argv[5], "--source-fd") != 0 ||
      strcmp(argv[7], "--pdf-output") != 0 || strcmp(argv[8], "stdout") != 0 ||
      !parse_source_descriptor(argv[6], &source_descriptor)) {
    return 64;
  }
  if (!verify_source_descriptor(source_descriptor)) {
    return 65;
  }
  if (chdir("/") != 0) {
    return 66;
  }
  if (!pin_dependency(DIALECT_RUNTIME_PATH, DIALECT_RUNTIME_SHA256,
                      (off_t)DIALECT_RUNTIME_BYTES,
                      (mode_t)DIALECT_RUNTIME_MODE, 1, &runtime_descriptor,
                      &runtime_identity)) {
    return 67;
  }
  if (!pin_dependency(DIALECT_RENDERER_PATH, DIALECT_RENDERER_SHA256,
                      (off_t)DIALECT_RENDERER_BYTES,
                      (mode_t)DIALECT_RENDERER_MODE, 0, &renderer_descriptor,
                      &renderer_identity)) {
    (void)close(runtime_descriptor);
    return 68;
  }
  result = prepare_verified_handoff(source_descriptor, runtime_descriptor,
                                    &runtime_identity, renderer_descriptor,
                                    &renderer_identity, argv, &handoff);
  if (result == 0) {
    result = spawn_attested_runtime(source_descriptor, runtime_descriptor,
                                    &runtime_identity, renderer_descriptor,
                                    &renderer_identity, &handoff);
  }
  (void)close(renderer_descriptor);
  (void)close(runtime_descriptor);
  return result;
}
