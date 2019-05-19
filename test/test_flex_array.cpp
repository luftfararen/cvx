#include "pch.h"
#include "../flex_array.h"

//using namespace cvx;

TEST(test_flex_array, constructor0) {
	flex_array<int,10> fr;
  EXPECT_EQ(fr.size(),0);
  EXPECT_EQ(fr.allocated(), false);
}

TEST(test_flex_array, constructor1)
{
  flex_array<int, 10> fr(5);
  EXPECT_EQ(fr.size(), 5);
  EXPECT_EQ(fr.allocated(), false);
}

TEST(test_flex_array, constructor2)
{
  flex_array<int, 10> fr(15);
  EXPECT_EQ(fr.size(), 15);
  EXPECT_EQ(fr.allocated(), true);
}

TEST(test_flex_array, resize0)
{
  flex_array<int, 10> fr;
	fr.resize(9);
  EXPECT_EQ(fr.size(), 9);
  EXPECT_EQ(fr.allocated(), false);
}

TEST(test_flex_array, resize1)
{
  flex_array<int, 10> fr;
  fr.resize(10);
  EXPECT_EQ(fr.size(), 10);
  EXPECT_EQ(fr.allocated(), false);
}

TEST(test_flex_array, resize2)
{
  flex_array<int, 10> fr;
  fr.resize(11);
  EXPECT_EQ(fr.size(), 11);
  EXPECT_EQ(fr.allocated(), true);
}
