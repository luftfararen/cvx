#include "pch.h"
#include "../stx.h"

using namespace stx;

TEST(rdiv_test, basic) {
  EXPECT_EQ(rdiv(4,2), 2);
  EXPECT_EQ(rdiv(2, 4), 1);
  EXPECT_EQ(rdiv(-2, 4), -1);
  EXPECT_EQ(rdiv(2, -4), -1);

	EXPECT_EQ(rdiv(3, 2), 2);
	EXPECT_EQ(rdiv(2, 3), 1);
	EXPECT_EQ(rdiv(-2, 3), -1);
	EXPECT_EQ(rdiv(2, -3), -1);
}

TEST(rdiv_test, float)
{
  EXPECT_EQ(rdiv(4.f, 2.f), 4.f/2.f);
  EXPECT_EQ(rdiv(2.f, 4.f), 2.f/4.f);
  EXPECT_EQ(rdiv(-2.f, 4.f), -2.f/4.f);
  EXPECT_EQ(rdiv(2.f, -4.f), 2.f/-4.f);
}

TEST(rdiv_test, double)
{
  EXPECT_EQ(rdiv(4.0, 2.0), 4.0 / 2.0);
  EXPECT_EQ(rdiv(2.0, 4.0), 2.0 / 4.0);
  EXPECT_EQ(rdiv(-2.0, 4.0), -2.0 / 4.0);
  EXPECT_EQ(rdiv(2.0, -4.0), 2.0 / -4.0);
}

TEST(iround_test, basic)
{
  EXPECT_EQ(lround(1.1f), 1);
	EXPECT_EQ(iround(1.1f), 1);
}

TEST(iround_test, float)
{
	EXPECT_EQ(iround(1.f), 1);
  EXPECT_EQ(iround(1.4f), 1);
  EXPECT_EQ(iround(1.5f), 2);
  EXPECT_EQ(iround(-1.5f), -2);
  EXPECT_EQ(iround(-1.4f), -1);
}

TEST(iround_test, double)
{
	EXPECT_EQ(iround(1.0), 1);
	EXPECT_EQ(iround(1.4), 1);
	EXPECT_EQ(iround(1.5), 2);
	EXPECT_EQ(iround(-1.5), -2);
	EXPECT_EQ(iround(-1.4), -1);
}

TEST(iround_test, long_double)
{
	EXPECT_EQ(iround(1.0l), 1);
	EXPECT_EQ(iround(1.4l), 1);
	EXPECT_EQ(iround(1.5l), 2);
	EXPECT_EQ(iround(-1.5l), -2);
	EXPECT_EQ(iround(-1.4l), -1);
}


TEST(tround_down_test, float) 
{
	EXPECT_EQ(itrunc(1.f), 1);
  EXPECT_EQ(itrunc(1.4f), 1);
  EXPECT_EQ(itrunc(1.5f), 1);
  EXPECT_EQ(itrunc(-1.5f), -1);
  EXPECT_EQ(itrunc(-1.4f), -1);
}

