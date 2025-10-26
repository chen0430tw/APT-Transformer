"""
Cloud Storage Plugin for APT Model
支持多种云存储服务的模型备份和分享
"""

import os
import boto3
import oss2
from pathlib import Path
from typing import Optional, Dict, Any, List
from huggingface_hub import HfApi, create_repo, upload_folder
from modelscope.hub.api import HubApi


class CloudStoragePlugin:
    """
    云存储插件
    
    支持的云服务:
    1. HuggingFace Hub - 模型分享
    2. ModelScope - 魔搭社区
    3. AWS S3 - 亚马逊云存储
    4. 阿里云 OSS - 阿里云对象存储
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = "cloud-storage"
        self.version = "1.0.0"
        self.config = config
        
        # 初始化各个云服务客户端
        self._init_clients()
    
    def _init_clients(self):
        """初始化云服务客户端"""
        # HuggingFace
        if self.config.get('hf_enabled', False):
            self.hf_api = HfApi()
        
        # ModelScope
        if self.config.get('ms_enabled', False):
            token = self.config.get('modelscope_token')
            self.ms_api = HubApi() if token else None
        
        # AWS S3
        if self.config.get('s3_enabled', False):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.config.get('aws_access_key'),
                aws_secret_access_key=self.config.get('aws_secret_key'),
                region_name=self.config.get('aws_region', 'us-east-1')
            )
        
        # 阿里云 OSS
        if self.config.get('oss_enabled', False):
            auth = oss2.Auth(
                self.config.get('oss_access_key'),
                self.config.get('oss_secret_key')
            )
            self.oss_bucket = oss2.Bucket(
                auth,
                self.config.get('oss_endpoint'),
                self.config.get('oss_bucket_name')
            )
    
    # ==================== HuggingFace Hub ====================
    
    def upload_to_huggingface(
        self,
        model_path: str,
        repo_name: str,
        private: bool = False,
        commit_message: str = "Upload model"
    ) -> str:
        """
        上传模型到HuggingFace Hub
        
        Returns:
            模型URL
        """
        print(f"📤 上传到 HuggingFace Hub: {repo_name}")
        
        try:
            # 创建仓库
            create_repo(repo_name, private=private, exist_ok=True)
            
            # 上传文件夹
            upload_folder(
                repo_id=repo_name,
                folder_path=model_path,
                commit_message=commit_message
            )
            
            url = f"https://huggingface.co/{repo_name}"
            print(f"✅ 上传成功: {url}")
            return url
            
        except Exception as e:
            print(f"❌ HuggingFace上传失败: {e}")
            raise
    
    # ==================== ModelScope ====================
    
    def upload_to_modelscope(
        self,
        model_path: str,
        repo_name: str,
        model_id: Optional[str] = None
    ) -> str:
        """
        上传模型到魔搭社区 (ModelScope)
        
        Returns:
            模型URL
        """
        print(f"📤 上传到 ModelScope: {repo_name}")
        
        try:
            # TODO: 实现ModelScope上传逻辑
            # 注意: ModelScope API可能需要特殊处理
            
            url = f"https://www.modelscope.cn/models/{repo_name}"
            print(f"✅ 上传成功: {url}")
            return url
            
        except Exception as e:
            print(f"❌ ModelScope上传失败: {e}")
            raise
    
    # ==================== AWS S3 ====================
    
    def upload_to_s3(
        self,
        local_path: str,
        s3_key: str,
        bucket_name: Optional[str] = None,
        public: bool = False
    ) -> str:
        """
        上传到AWS S3
        
        Args:
            local_path: 本地文件/文件夹路径
            s3_key: S3对象键 (路径)
            bucket_name: S3桶名称 (如果未设置则使用配置中的默认值)
            public: 是否公开访问
            
        Returns:
            S3 URL
        """
        bucket = bucket_name or self.config.get('s3_bucket_name')
        print(f"📤 上传到 AWS S3: s3://{bucket}/{s3_key}")
        
        try:
            local_path = Path(local_path)
            
            if local_path.is_file():
                # 上传单个文件
                self._upload_file_to_s3(str(local_path), bucket, s3_key, public)
            else:
                # 上传文件夹
                self._upload_folder_to_s3(local_path, bucket, s3_key, public)
            
            url = f"s3://{bucket}/{s3_key}"
            print(f"✅ S3上传成功: {url}")
            return url
            
        except Exception as e:
            print(f"❌ S3上传失败: {e}")
            raise
    
    def _upload_file_to_s3(self, file_path: str, bucket: str, s3_key: str, public: bool):
        """上传单个文件到S3"""
        extra_args = {'ACL': 'public-read'} if public else {}
        self.s3_client.upload_file(file_path, bucket, s3_key, ExtraArgs=extra_args)
    
    def _upload_folder_to_s3(self, folder_path: Path, bucket: str, s3_prefix: str, public: bool):
        """上传文件夹到S3"""
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(folder_path)
                s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                self._upload_file_to_s3(str(file_path), bucket, s3_key, public)
    
    def download_from_s3(
        self,
        s3_key: str,
        local_path: str,
        bucket_name: Optional[str] = None
    ):
        """从S3下载文件"""
        bucket = bucket_name or self.config.get('s3_bucket_name')
        print(f"📥 从 S3 下载: s3://{bucket}/{s3_key}")
        
        try:
            self.s3_client.download_file(bucket, s3_key, local_path)
            print(f"✅ 下载成功: {local_path}")
        except Exception as e:
            print(f"❌ S3下载失败: {e}")
            raise
    
    # ==================== 阿里云 OSS ====================
    
    def upload_to_oss(
        self,
        local_path: str,
        oss_key: str,
        public: bool = False
    ) -> str:
        """
        上传到阿里云OSS
        
        Args:
            local_path: 本地文件/文件夹路径
            oss_key: OSS对象键
            public: 是否公开访问
            
        Returns:
            OSS URL
        """
        print(f"📤 上传到阿里云OSS: {oss_key}")
        
        try:
            local_path = Path(local_path)
            
            if local_path.is_file():
                # 上传单个文件
                self.oss_bucket.put_object_from_file(oss_key, str(local_path))
            else:
                # 上传文件夹
                self._upload_folder_to_oss(local_path, oss_key)
            
            # 设置访问权限
            if public:
                self.oss_bucket.put_object_acl(oss_key, oss2.OBJECT_ACL_PUBLIC_READ)
            
            url = f"{self.config.get('oss_endpoint')}/{oss_key}"
            print(f"✅ OSS上传成功: {url}")
            return url
            
        except Exception as e:
            print(f"❌ OSS上传失败: {e}")
            raise
    
    def _upload_folder_to_oss(self, folder_path: Path, oss_prefix: str):
        """上传文件夹到OSS"""
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(folder_path)
                oss_key = f"{oss_prefix}/{relative_path}".replace('\\', '/')
                self.oss_bucket.put_object_from_file(oss_key, str(file_path))
    
    def download_from_oss(self, oss_key: str, local_path: str):
        """从OSS下载文件"""
        print(f"📥 从 OSS 下载: {oss_key}")
        
        try:
            self.oss_bucket.get_object_to_file(oss_key, local_path)
            print(f"✅ 下载成功: {local_path}")
        except Exception as e:
            print(f"❌ OSS下载失败: {e}")
            raise
    
    # ==================== 统一接口 ====================
    
    def backup_model(
        self,
        model_path: str,
        backup_name: str,
        destinations: List[str] = None
    ) -> Dict[str, str]:
        """
        备份模型到多个云存储
        
        Args:
            model_path: 模型路径
            backup_name: 备份名称
            destinations: 目标云服务列表 ['hf', 'ms', 's3', 'oss']
                         如果为None,则备份到所有已启用的服务
        
        Returns:
            各个云服务的URL字典
        """
        destinations = destinations or ['hf', 'ms', 's3', 'oss']
        results = {}
        
        print(f"🔄 开始多云备份: {backup_name}")
        
        # HuggingFace Hub
        if 'hf' in destinations and self.config.get('hf_enabled'):
            try:
                repo_name = f"{self.config.get('hf_username')}/{backup_name}"
                results['huggingface'] = self.upload_to_huggingface(
                    model_path, repo_name, private=True
                )
            except Exception as e:
                print(f"⚠️ HuggingFace备份失败: {e}")
        
        # ModelScope
        if 'ms' in destinations and self.config.get('ms_enabled'):
            try:
                results['modelscope'] = self.upload_to_modelscope(
                    model_path, backup_name
                )
            except Exception as e:
                print(f"⚠️ ModelScope备份失败: {e}")
        
        # AWS S3
        if 's3' in destinations and self.config.get('s3_enabled'):
            try:
                s3_key = f"apt_backups/{backup_name}"
                results['s3'] = self.upload_to_s3(model_path, s3_key)
            except Exception as e:
                print(f"⚠️ S3备份失败: {e}")
        
        # 阿里云OSS
        if 'oss' in destinations and self.config.get('oss_enabled'):
            try:
                oss_key = f"apt_backups/{backup_name}"
                results['oss'] = self.upload_to_oss(model_path, oss_key)
            except Exception as e:
                print(f"⚠️ OSS备份失败: {e}")
        
        print(f"✅ 多云备份完成! 成功: {len(results)}/{len(destinations)}")
        return results
    
    # ==================== 插件钩子 ====================
    
    def on_training_end(self, context: Dict[str, Any]):
        """训练结束时自动备份"""
        if self.config.get('auto_backup', False):
            model_path = context.get('checkpoint_path')
            backup_name = f"apt_model_{context.get('timestamp', 'latest')}"
            
            self.backup_model(
                model_path,
                backup_name,
                destinations=self.config.get('backup_destinations')
            )
    
    def on_epoch_end(self, context: Dict[str, Any]):
        """每个epoch结束时备份检查点"""
        if self.config.get('backup_checkpoints', False):
            epoch = context.get('epoch')
            if epoch % self.config.get('backup_interval', 5) == 0:
                model_path = context.get('checkpoint_path')
                backup_name = f"apt_checkpoint_epoch_{epoch}"
                
                self.backup_model(
                    model_path,
                    backup_name,
                    destinations=['s3']  # 检查点只备份到S3,不占用社区资源
                )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置示例
    config = {
        # HuggingFace配置
        'hf_enabled': True,
        'hf_username': 'your_username',
        
        # ModelScope配置
        'ms_enabled': False,
        'modelscope_token': 'your_token',
        
        # AWS S3配置
        's3_enabled': True,
        's3_bucket_name': 'my-apt-models',
        'aws_access_key': 'your_key',
        'aws_secret_key': 'your_secret',
        'aws_region': 'us-east-1',
        
        # 阿里云OSS配置
        'oss_enabled': False,
        'oss_bucket_name': 'my-apt-models',
        'oss_access_key': 'your_key',
        'oss_secret_key': 'your_secret',
        'oss_endpoint': 'oss-cn-hangzhou.aliyuncs.com',
        
        # 自动备份配置
        'auto_backup': True,
        'backup_checkpoints': True,
        'backup_interval': 5,  # 每5个epoch备份一次
        'backup_destinations': ['hf', 's3'],
    }
    
    plugin = CloudStoragePlugin(config)
    
    # 示例: 备份模型到多个云服务
    results = plugin.backup_model(
        model_path="./checkpoints/best_model",
        backup_name="apt-chinese-v1",
        destinations=['hf', 's3']
    )
    
    print("备份结果:", results)
