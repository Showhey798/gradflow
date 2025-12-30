#include <gradflow/autograd/allocator.hpp>
#include <gradflow/autograd/device.hpp>
#include <gradflow/autograd/storage.hpp>
#include <gtest/gtest.h>

namespace gradflow {
namespace test {

// ========================================
// Device Tests
// ========================================

TEST(DeviceTest, Construction) {
    Device cpu_device(DeviceType::CPU, 0);
    EXPECT_EQ(cpu_device.type(), DeviceType::CPU);
    EXPECT_EQ(cpu_device.index(), 0);
    EXPECT_TRUE(cpu_device.is_cpu());
    EXPECT_FALSE(cpu_device.is_cuda());
    EXPECT_FALSE(cpu_device.is_metal());

    Device cuda_device(DeviceType::CUDA, 1);
    EXPECT_EQ(cuda_device.type(), DeviceType::CUDA);
    EXPECT_EQ(cuda_device.index(), 1);
    EXPECT_FALSE(cuda_device.is_cpu());
    EXPECT_TRUE(cuda_device.is_cuda());
    EXPECT_FALSE(cuda_device.is_metal());
}

TEST(DeviceTest, FactoryFunctions) {
    Device cpu_device = cpu();
    EXPECT_TRUE(cpu_device.is_cpu());
    EXPECT_EQ(cpu_device.index(), 0);

    Device cuda_device = cuda(2);
    EXPECT_TRUE(cuda_device.is_cuda());
    EXPECT_EQ(cuda_device.index(), 2);

    Device metal_device = metal(1);
    EXPECT_TRUE(metal_device.is_metal());
    EXPECT_EQ(metal_device.index(), 1);
}

TEST(DeviceTest, Equality) {
    Device cpu1 = cpu();
    Device cpu2 = cpu();
    Device cuda1 = cuda(0);
    Device cuda2 = cuda(1);

    EXPECT_EQ(cpu1, cpu2);
    EXPECT_NE(cpu1, cuda1);
    EXPECT_NE(cuda1, cuda2);
}

TEST(DeviceTest, ToString) {
    Device cpu_device = cpu();
    EXPECT_EQ(cpu_device.to_string(), "cpu:0");

    Device cuda_device = cuda(1);
    EXPECT_EQ(cuda_device.to_string(), "cuda:1");

    Device metal_device = metal(2);
    EXPECT_EQ(metal_device.to_string(), "metal:2");
}

// ========================================
// CPUAllocator Tests
// ========================================

TEST(CPUAllocatorTest, Construction) {
    CPUAllocator allocator;
    EXPECT_EQ(allocator.device(), cpu());
    EXPECT_EQ(allocator.alignment(), CPUAllocator::DEFAULT_ALIGNMENT);
}

TEST(CPUAllocatorTest, CustomAlignment) {
    CPUAllocator allocator(32);
    EXPECT_EQ(allocator.alignment(), 32);

    // Invalid alignment (not power of 2)
    EXPECT_THROW(CPUAllocator(33), std::invalid_argument);
    EXPECT_THROW(CPUAllocator(0), std::invalid_argument);
}

TEST(CPUAllocatorTest, AlignedAllocation) {
    CPUAllocator allocator(64);

    // Allocate memory
    void* ptr = allocator.allocate(100);
    ASSERT_NE(ptr, nullptr);

    // Check alignment
    uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(address % 64, 0) << "Memory is not aligned to 64 bytes";

    // Deallocate
    allocator.deallocate(ptr);
}

TEST(CPUAllocatorTest, ZeroSizeAllocation) {
    CPUAllocator allocator;

    // Allocating zero bytes should return nullptr
    void* ptr = allocator.allocate(0);
    EXPECT_EQ(ptr, nullptr);

    // Deallocating nullptr should be safe
    allocator.deallocate(nullptr);
}

TEST(CPUAllocatorTest, MultipleAllocations) {
    CPUAllocator allocator;

    // Allocate multiple blocks
    void* ptr1 = allocator.allocate(100);
    void* ptr2 = allocator.allocate(200);
    void* ptr3 = allocator.allocate(300);

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr3, nullptr);

    // Pointers should be different
    EXPECT_NE(ptr1, ptr2);
    EXPECT_NE(ptr2, ptr3);
    EXPECT_NE(ptr1, ptr3);

    // Deallocate all
    allocator.deallocate(ptr1);
    allocator.deallocate(ptr2);
    allocator.deallocate(ptr3);
}

TEST(CPUAllocatorTest, GetDefaultAllocator) {
    auto allocator1 = get_default_cpu_allocator();
    auto allocator2 = get_default_cpu_allocator();

    // Should return the same instance (singleton pattern)
    EXPECT_EQ(allocator1, allocator2);
    EXPECT_EQ(allocator1->device(), cpu());
}

// ========================================
// Storage Tests
// ========================================

TEST(StorageTest, Construction) {
    // Empty storage
    Storage<float> storage1;
    EXPECT_EQ(storage1.size(), 0);
    EXPECT_TRUE(storage1.empty());
    EXPECT_EQ(storage1.data(), nullptr);

    // Storage with size
    Storage<float> storage2(10);
    EXPECT_EQ(storage2.size(), 10);
    EXPECT_FALSE(storage2.empty());
    EXPECT_NE(storage2.data(), nullptr);
}

TEST(StorageTest, Allocation) {
    // Allocate storage with default allocator
    Storage<double> storage(100);
    EXPECT_EQ(storage.size(), 100);
    EXPECT_NE(storage.data(), nullptr);
    EXPECT_EQ(storage.device(), cpu());

    // Allocate storage with custom allocator
    auto allocator = std::make_shared<CPUAllocator>(32);
    Storage<int> storage_custom(50, allocator);
    EXPECT_EQ(storage_custom.size(), 50);
    EXPECT_NE(storage_custom.data(), nullptr);
    EXPECT_EQ(storage_custom.allocator(), allocator);
}

TEST(StorageTest, DataAccess) {
    Storage<int> storage(10);

    // Write data
    for (size_t i = 0; i < storage.size(); ++i) {
        storage[i] = static_cast<int>(i * 2);
    }

    // Read data
    for (size_t i = 0; i < storage.size(); ++i) {
        EXPECT_EQ(storage[i], static_cast<int>(i * 2));
    }

    // Test const access
    const Storage<int>& const_storage = storage;
    for (size_t i = 0; i < const_storage.size(); ++i) {
        EXPECT_EQ(const_storage[i], static_cast<int>(i * 2));
    }
}

TEST(StorageTest, BoundsChecking) {
    Storage<float> storage(5);

    // Valid access
    EXPECT_NO_THROW(storage.at(0));
    EXPECT_NO_THROW(storage.at(4));

    // Invalid access
    EXPECT_THROW(storage.at(5), std::out_of_range);
    EXPECT_THROW(storage.at(100), std::out_of_range);
}

TEST(StorageTest, Fill) {
    Storage<float> storage(10);

    // Fill with value
    storage.fill(3.14F);

    // Verify all elements
    for (size_t i = 0; i < storage.size(); ++i) {
        EXPECT_FLOAT_EQ(storage[i], 3.14F);
    }
}

TEST(StorageTest, MoveSemantics) {
    // Create storage
    Storage<int> storage1(10);
    storage1.fill(42);
    int* original_ptr = storage1.data();

    // Move construct
    Storage<int> storage2(std::move(storage1));
    EXPECT_EQ(storage2.size(), 10);
    EXPECT_EQ(storage2.data(), original_ptr);
    EXPECT_EQ(storage2[0], 42);

    // Original storage should be empty
    EXPECT_EQ(storage1.size(), 0);        // NOLINT
    EXPECT_EQ(storage1.data(), nullptr);  // NOLINT

    // Move assign
    Storage<int> storage3(5);
    storage3 = std::move(storage2);
    EXPECT_EQ(storage3.size(), 10);
    EXPECT_EQ(storage3.data(), original_ptr);
    EXPECT_EQ(storage3[0], 42);

    // storage2 should be empty
    EXPECT_EQ(storage2.size(), 0);        // NOLINT
    EXPECT_EQ(storage2.data(), nullptr);  // NOLINT
}

TEST(StorageTest, CopyFrom) {
    Storage<float> storage1(10);
    Storage<float> storage2(10);

    // Fill storage1 with data
    for (size_t i = 0; i < storage1.size(); ++i) {
        storage1[i] = static_cast<float>(i);
    }

    // Copy from storage1 to storage2
    storage2.copy_from(storage1);

    // Verify data
    for (size_t i = 0; i < storage2.size(); ++i) {
        EXPECT_FLOAT_EQ(storage2[i], static_cast<float>(i));
    }

    // Modifying storage2 should not affect storage1
    storage2[0] = 999.0F;
    EXPECT_FLOAT_EQ(storage1[0], 0.0F);
    EXPECT_FLOAT_EQ(storage2[0], 999.0F);
}

TEST(StorageTest, CopyFromSizeMismatch) {
    Storage<int> storage1(10);
    Storage<int> storage2(5);

    // Copy should throw when sizes don't match
    EXPECT_THROW(storage2.copy_from(storage1), std::invalid_argument);
}

TEST(StorageTest, ZeroSizeStorage) {
    Storage<float> storage(0);

    EXPECT_EQ(storage.size(), 0);
    EXPECT_TRUE(storage.empty());
    EXPECT_EQ(storage.data(), nullptr);

    // Fill should be safe on empty storage
    EXPECT_NO_THROW(storage.fill(1.0F));
}

TEST(StorageTest, LargeAllocation) {
    // Test allocation of large storage (10MB)
    constexpr size_t large_size = 10 * 1024 * 1024 / sizeof(float);
    Storage<float> storage(large_size);

    EXPECT_EQ(storage.size(), large_size);
    EXPECT_NE(storage.data(), nullptr);

    // Write and read a few values
    storage[0] = 1.0F;
    storage[large_size / 2] = 2.0F;
    storage[large_size - 1] = 3.0F;

    EXPECT_FLOAT_EQ(storage[0], 1.0F);
    EXPECT_FLOAT_EQ(storage[large_size / 2], 2.0F);
    EXPECT_FLOAT_EQ(storage[large_size - 1], 3.0F);
}

// ========================================
// RAII and Memory Leak Tests
// ========================================

TEST(StorageTest, RAIIMemoryManagement) {
    auto allocator = std::make_shared<CPUAllocator>();

    // Create and destroy storage in a scope
    {
        Storage<double> storage(100, allocator);
        storage.fill(1.0);
    }
    // Storage destructor should have been called here

    // If we reach here without crash, RAII worked correctly
    SUCCEED();
}

TEST(StorageTest, ExceptionSafety) {
    // Even if an exception is thrown, storage should be properly cleaned up
    try {
        Storage<int> storage(10);
        storage.fill(42);
        throw std::runtime_error("Test exception");
    } catch (const std::runtime_error&) {
        // Storage destructor should have been called
        SUCCEED();
    }
}

}  // namespace test
}  // namespace gradflow
