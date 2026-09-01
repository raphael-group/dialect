#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef DIALECT_RUNTIME_PATH
#error DIALECT_RUNTIME_PATH must be supplied by the authority builder
#endif
#ifndef DIALECT_RUNTIME_SHA256
#error DIALECT_RUNTIME_SHA256 must be supplied by the authority builder
#endif
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
#define DIALECT_ROLE "rebuttal"
#define DIALECT_MAX_BUNDLE_BYTES (8LL * 1024LL * 1024LL)
#define DIALECT_IO_CHUNK 65536U

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
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
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
    words[index] = words[index - 16U] + small_zero + words[index - 7U] +
                   small_one;
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
    digest[index * 4U + 1U] =
        (unsigned char)(context->state[index] >> 16U);
    digest[index * 4U + 2U] =
        (unsigned char)(context->state[index] >> 8U);
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

static int digest_matches(const unsigned char digest[32], const char *expected) {
  size_t index;
  if (strlen(expected) != 64U) {
    return 0;
  }
  for (index = 0U; index < 32U; ++index) {
    const int high = hex_value(expected[index * 2U]);
    const int low = hex_value(expected[index * 2U + 1U]);
    if (high < 0 || low < 0 || digest[index] != (unsigned char)((high << 4) | low)) {
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
         first->st_mode == second->st_mode && first->st_nlink == second->st_nlink &&
         first->st_uid == second->st_uid;
}

static int verify_dependency(const char *path, const char *expected_sha256,
                             off_t expected_size, mode_t expected_mode,
                             int executable) {
  unsigned char buffer[DIALECT_IO_CHUNK];
  unsigned char digest[32];
  struct dialect_sha256 hash;
  struct stat before;
  struct stat after;
  off_t offset = 0;
  int descriptor;

  descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0 || fstat(descriptor, &before) != 0) {
    if (descriptor >= 0) {
      (void)close(descriptor);
    }
    return 0;
  }
  if (!S_ISREG(before.st_mode) || before.st_nlink != 1 ||
      before.st_uid != geteuid() || before.st_size != expected_size ||
      (before.st_mode & 07777) != expected_mode ||
      (executable != 0 && (before.st_mode & S_IXUSR) == 0)) {
    (void)close(descriptor);
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
      (void)close(descriptor);
      return 0;
    }
    sha256_update(&hash, buffer, (size_t)count);
    offset += count;
  }
  sha256_final(&hash, digest);
  if (fstat(descriptor, &after) != 0 || !stat_identity_matches(&before, &after) ||
      !digest_matches(digest, expected_sha256) || close(descriptor) != 0) {
    return 0;
  }
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
      before.st_nlink != 1 || before.st_uid != geteuid() || before.st_size < 1 ||
      before.st_size > DIALECT_MAX_BUNDLE_BYTES ||
      (before.st_mode & 07777) != 0400 ||
      lseek(descriptor, 0, SEEK_CUR) < 0 || lseek(descriptor, 0, SEEK_SET) != 0 ||
      fstat(descriptor, &after) != 0 || !stat_identity_matches(&before, &after)) {
    return 0;
  }
  return 1;
}

static int clear_source_cloexec(int descriptor) {
  int descriptor_flags = fcntl(descriptor, F_GETFD);
  if (descriptor_flags < 0 ||
      fcntl(descriptor, F_SETFD, descriptor_flags & ~FD_CLOEXEC) != 0 ||
      (fcntl(descriptor, F_GETFD) & FD_CLOEXEC) != 0) {
    return 0;
  }
  return 1;
}

static int close_unexpected_descriptors(int source_descriptor) {
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
    if (descriptor > 2 && descriptor != source_descriptor &&
        descriptor != directory_descriptor) {
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

int main(int argc, char *argv[]) {
  int source_descriptor;
  char *child_argv[14];
  char *environment[4];

  if (argc != 9 || strcmp(argv[1], "--dialect-derivation-protocol") != 0 ||
      strcmp(argv[2], DIALECT_PROTOCOL) != 0 ||
      strcmp(argv[3], "--pdf-id") != 0 || strcmp(argv[4], DIALECT_ROLE) != 0 ||
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
  if (!verify_dependency(DIALECT_RUNTIME_PATH, DIALECT_RUNTIME_SHA256,
                         (off_t)DIALECT_RUNTIME_BYTES,
                         (mode_t)DIALECT_RUNTIME_MODE, 1)) {
    return 67;
  }
  if (!verify_dependency(DIALECT_RENDERER_PATH, DIALECT_RENDERER_SHA256,
                         (off_t)DIALECT_RENDERER_BYTES,
                         (mode_t)DIALECT_RENDERER_MODE, 0)) {
    return 68;
  }
  if (!close_unexpected_descriptors(source_descriptor)) {
    return 69;
  }
  if (!clear_source_cloexec(source_descriptor)) {
    return 65;
  }

  child_argv[0] = (char *)DIALECT_RUNTIME_PATH;
  child_argv[1] = (char *)"-I";
  child_argv[2] = (char *)"-S";
  child_argv[3] = (char *)"-B";
  child_argv[4] = (char *)DIALECT_RENDERER_PATH;
  child_argv[5] = argv[1];
  child_argv[6] = argv[2];
  child_argv[7] = argv[3];
  child_argv[8] = argv[4];
  child_argv[9] = argv[5];
  child_argv[10] = argv[6];
  child_argv[11] = argv[7];
  child_argv[12] = argv[8];
  child_argv[13] = NULL;
  environment[0] = (char *)"LANG=C";
  environment[1] = (char *)"LC_ALL=C";
  environment[2] = (char *)"TZ=UTC";
  environment[3] = NULL;
  execve(DIALECT_RUNTIME_PATH, child_argv, environment);
  return 126;
}
