-- 1. Create the Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE, -- Links to the logged-in user
    amount DECIMAL(15, 2) NOT NULL,
    tx_type TEXT NOT NULL,
    hour INTEGER NOT NULL,
    fraud_probability INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    confidence TEXT NOT NULL,
    reasoning TEXT,
    model_used TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Enable Row Level Security (RLS)
-- This ensures users can only see their OWN transactions
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;

-- 3. Create a policy: Allow users to select only their own rows
CREATE POLICY "Users can view their own transactions" 
ON transactions FOR SELECT 
USING (auth.uid() = user_id);

-- 4. Create a policy: Allow users to insert their own rows
CREATE POLICY "Users can insert their own transactions" 
ON transactions FOR INSERT 
WITH CHECK (auth.uid() = user_id);

-- 5. Create an index on created_at for faster dashboard loading
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC);

-- 6. Create an index on user_id for faster filtering
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);

COMMENT ON TABLE transactions IS 'Stores fraud analysis results for the FraudShield dashboard.';
